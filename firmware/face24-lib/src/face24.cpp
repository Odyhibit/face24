#pragma GCC optimize("O3")
#include "face24.h"

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <soc/soc_memory_layout.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>
#include <dsps_dotprod.h>
#include <math.h>
#include <string.h>

#include <memory>
#include <string>
#include <stdio.h>
#include <vector>

#ifndef FACE24_PROFILE
#define FACE24_PROFILE 0  // 1: print per-layer timings to USB serial
#endif

namespace face24 {
namespace {

struct Header {
    char magic[4];
    uint32_t version, payload_bytes, kind, grid_h, grid_w, n_tensors, n_ops, codebook_a, codebook_b, data_offset,
        reserved;
};
struct TensorDesc {
    uint32_t dtype, ndim, shape[4], offset, scale_offset, scale_axis;
};
struct OpDesc {
    int32_t type, w, b, sh, sw, ph, pw, size;
};
static_assert(sizeof(Header) == 48, "header layout");
static_assert(sizeof(TensorDesc) == 36, "tensor layout");
static_assert(sizeof(OpDesc) == 32, "op layout");

enum { OP_CONV = 1, OP_CONVT, OP_RELU, OP_SIGMOID, OP_UP2, OP_UPSIZE, OP_RES_BEGIN, OP_RES_END };
enum { DT_F32 = 0, DT_I8 = 1 };

struct Entry {
    uint32_t payload_bytes, offset, size;
};

const esp_partition_t* g_part = nullptr;
std::vector<Entry> g_entries;
int g_mapped = -1;
spi_flash_mmap_handle_t g_handle;
const uint8_t* g_base = nullptr;  // start of the mapped model
const Header* g_hdr   = nullptr;
const TensorDesc* g_tensors = nullptr;
const OpDesc* g_ops         = nullptr;
const uint8_t* g_data       = nullptr;
const char* g_error         = "";
Stats g_stats;

// Per-output-channel accumulator, kept in internal RAM (largest plane is 80x64).
float g_acc[IMAGE_H * IMAGE_W];

/* ------------------------------------------------------------- tensors */

void freeBuf(float* p)
{
    heap_caps_free(p);
}

struct Tensor {
    int c = 0, h = 0, w = 0;
    std::shared_ptr<float> buf;
    float* data() const
    {
        return buf.get();
    }
    size_t size() const
    {
        return (size_t)c * h * w;
    }
};

Tensor makeTensor(int c, int h, int w)
{
    Tensor t;
    t.c = c;
    t.h = h;
    t.w = w;
    size_t bytes = t.size() * sizeof(float);
    void* p      = nullptr;
    if (bytes <= 48 * 1024) p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p) t.buf = std::shared_ptr<float>((float*)p, freeBuf);
    return t;
}

// Returns x itself if nothing else holds it, otherwise a private copy.
Tensor writable(Tensor x)
{
    if (x.buf.use_count() <= 2) return x;  // x and the caller's reference
    Tensor y = makeTensor(x.c, x.h, x.w);
    if (y.buf) memcpy(y.data(), x.data(), x.size() * sizeof(float));
    return y;
}

template <typename T>
const T* at(uint32_t offset)
{
    return reinterpret_cast<const T*>(g_data + offset);
}

/* -------------------------------------------------------------- layers */

Tensor conv(const Tensor& x, const OpDesc& op)
{
    const TensorDesc& wd = g_tensors[op.w];
    const TensorDesc& bd = g_tensors[op.b];
    const int O = wd.shape[0], I = wd.shape[1], KH = wd.shape[2], KW = wd.shape[3];
    const int sh = op.sh, sw = op.sw, ph = op.ph, pw = op.pw;
    const int H = (x.h + 2 * ph - KH) / sh + 1;
    const int W = (x.w + 2 * pw - KW) / sw + 1;
    Tensor y    = makeTensor(O, H, W);
    if (!y.buf || I != x.c || H * W > IMAGE_H * IMAGE_W) return Tensor();

    const int8_t* q    = at<int8_t>(wd.offset);
    const float* scale = at<float>(wd.scale_offset);
    const float* bias  = at<float>(bd.offset);

    // Valid output ranges per kernel offset, so the inner loop has no bounds checks.
    int xr0[8], xr1[8], yr0[8], yr1[8];
    for (int kx = 0; kx < KW; kx++) {
        xr0[kx] = max(0, (int)ceilf((float)(pw - kx) / sw));
        xr1[kx] = min(W, (x.w - 1 + pw - kx) / sw + 1);
    }
    for (int ky = 0; ky < KH; ky++) {
        yr0[ky] = max(0, (int)ceilf((float)(ph - ky) / sh));
        yr1[ky] = min(H, (x.h - 1 + ph - ky) / sh + 1);
    }

    const int plane = x.h * x.w, HW = H * W;
    for (int o = 0; o < O; o++) {
        float* acc = g_acc;
        for (int n = 0; n < HW; n++) acc[n] = bias[o];
        const float s     = scale[o];
        const int8_t* qo  = q + (size_t)o * I * KH * KW;
        for (int i = 0; i < I; i++) {
            const float* src = x.data() + (size_t)i * plane;
            for (int ky = 0; ky < KH; ky++) {
                for (int kx = 0; kx < KW; kx++) {
                    const int8_t qv = qo[(i * KH + ky) * KW + kx];
                    if (qv == 0) continue;
                    const float k = qv * s;
                    const int x0 = xr0[kx], x1 = xr1[kx], n = x1 - x0;
                    for (int yy = yr0[ky]; yy < yr1[ky]; yy++) {
                        const float* ip = src + (yy * sh + ky - ph) * x.w + x0 * sw + kx - pw;
                        float* op_      = acc + yy * W + x0;
                        if (sw == 1) {
                            for (int j = 0; j < n; j++) op_[j] += k * ip[j];
                        } else {
                            for (int j = 0; j < n; j++) op_[j] += k * ip[j * sw];
                        }
                    }
                }
            }
        }
        memcpy(y.data() + (size_t)o * HW, acc, HW * sizeof(float));
    }
    return y;
}

#if CONFIG_IDF_TARGET_ESP32S3
// ESP32-S3: PIE vector instructions. Each ee.vmulas.s16.accx multiplies 8 int16 pairs and adds
// them to the 40-bit ACCX accumulator. n must be a multiple of 16 and both arrays 16-byte aligned
// (ee.vld.128 ignores the low 4 address bits). No reads past the end.
#define FACE24_DOT_MULTIPLE 16  // two 8-value vector steps per loop half
static inline int64_t dot_s16(const int16_t* a, const int16_t* b, int n)
{
    uint32_t lo;
    uint32_t hi;
    const int16_t* pa = a;
    const int16_t* pb = b;
    // Software-pipelined: each multiply also loads the next vector of a (the ".ld.ip" form),
    // alternating q0/q1 and q2/q3, so loads overlap the multiplies. n/8 must be even (every
    // decoder row is a multiple of 64 values); the last pair is done after the loop.
    const int half_minus_1 = (n >> 4) - 1;
    asm volatile(
        "ee.zero.accx\n"
        "ee.vld.128.ip q0, %[pa], 16\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "loopnez %[it], 1f\n"
        "ee.vmulas.s16.accx.ld.ip q2, %[pa], 16, q0, q1\n"
        "ee.vld.128.ip q3, %[pb], 16\n"
        "ee.vmulas.s16.accx.ld.ip q0, %[pa], 16, q2, q3\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "1:\n"
        "ee.vmulas.s16.accx.ld.ip q2, %[pa], 16, q0, q1\n"
        "ee.vld.128.ip q3, %[pb], 16\n"
        "ee.vmulas.s16.accx q2, q3\n"
        "rur.accx_0 %[lo]\n"
        "rur.accx_1 %[hi]\n"
        : [pa] "+r"(pa), [pb] "+r"(pb), [lo] "=&r"(lo), [hi] "=&r"(hi)
        : [it] "r"(half_minus_1)
        : "memory");
    return (int64_t)((uint64_t)(int64_t)(int8_t)(hi & 0xFF) << 32) | lo;
}
#else
// ESP32: MAC16 unit (40-bit accumulator).
// n must be a multiple of 4 and both arrays 4-byte aligned. The loop loads one
// word (2 values) ahead, so both arrays need 4 readable bytes past the end.
#define FACE24_DOT_MULTIPLE 4
static inline int64_t dot_s16(const int16_t* a, const int16_t* b, int n)
{
    uint32_t lo;
    uint32_t hi;
    const int16_t* pa = a - 2;  // ldinc pre-increments by 4 bytes
    const int16_t* pb = b - 2;
    const int iters   = n >> 2;
    const int zero    = 0;
    asm volatile(
        "wsr    %[z], acclo\n"
        "wsr    %[z], acchi\n"
        "ldinc  m0, %[pa]\n"
        "ldinc  m2, %[pb]\n"
        "loopnez %[it], 1f\n"
        "mula.dd.ll.ldinc m1, %[pa], m0, m2\n"
        "mula.dd.hh.ldinc m3, %[pb], m0, m2\n"
        "mula.dd.ll.ldinc m0, %[pa], m1, m3\n"
        "mula.dd.hh.ldinc m2, %[pb], m1, m3\n"
        "1:\n"
        "rsr    %[lo], acclo\n"
        "rsr    %[hi], acchi\n"
        : [pa] "+r"(pa), [pb] "+r"(pb), [lo] "=&r"(lo), [hi] "=&r"(hi)
        : [it] "r"(iters), [z] "r"(zero)
        : "memory");
    return (int64_t)((uint64_t)(int64_t)(int8_t)(hi & 0xFF) << 32) | lo;
}
#endif

// Stride-1 convolution with int16 activations and int8 weights:
//   - input quantized to int16 (one scale per tensor, from its max |value|),
//     stored channel-last (HWC) with a zero border, so the KW pixels of a kernel
//     row are KW*I contiguous values: no patch gathering,
//   - weights for a block of output channels sign-extended to int16 in internal RAM,
//     reordered to [ky][kx][i] to match,
//   - dot products on the MAC16 unit; output = bias + sum * (x_scale * w_scale[o]).
Tensor convDot(const Tensor& x, const OpDesc& op)
{
    const TensorDesc& wd = g_tensors[op.w];
    const TensorDesc& bd = g_tensors[op.b];
    const int O = wd.shape[0], I = wd.shape[1], KH = wd.shape[2], KW = wd.shape[3];
    const int ph = op.ph, pw = op.pw;
    const int H = x.h + 2 * ph - KH + 1, W = x.w + 2 * pw - KW + 1;
    const int K = I * KH * KW, HW = H * W, plane = x.h * x.w;
    const int row_len = KW * I;
    if (I != x.c || row_len % FACE24_DOT_MULTIPLE != 0) return conv(x, op);
    Tensor y = makeTensor(O, H, W);
    if (!y.buf) return Tensor();

    size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int B = (int)min<size_t>(32, (largest > 24 * 1024 ? largest - 16 * 1024 : 0) / (K * sizeof(int16_t)));
    if (B < 1) return conv(x, op);
    // Every row the dot product reads starts a multiple of 128 bytes into these buffers
    // (I is 64, 128 or 256), so aligning the buffers aligns every row.
    int16_t* wq = (int16_t*)heap_caps_aligned_alloc(16, (size_t)B * K * sizeof(int16_t) + 16, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const int Hp = x.h + 2 * ph, Wp = x.w + 2 * pw;
    const size_t xp_bytes = (size_t)Hp * Wp * I * sizeof(int16_t) + 16;
    int16_t* xp = (int16_t*)heap_caps_aligned_alloc(16, xp_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (xp) memset(xp, 0, xp_bytes);
    if (!wq || !xp) {
        heap_caps_free(wq);
        heap_caps_free(xp);
        return conv(x, op);
    }

    const int8_t* q    = at<int8_t>(wd.offset);
    const float* scale = at<float>(wd.scale_offset);
    const float* bias  = at<float>(bd.offset);
    const float* in    = x.data();
    float* out         = y.data();
    uint32_t t_w = 0, t_g = 0, t_d = 0, t0 = micros(), t1;

    // Quantize + transpose the input.
    float maxabs = 0;
    for (size_t n = 0, N = x.size(); n < N; n++) maxabs = fmaxf(maxabs, fabsf(in[n]));
    const float x_scale = maxabs > 0 ? maxabs / 32000.0f : 1.0f;
    const float inv     = 1.0f / x_scale;
    // Row by row, so the destination row (W*I values) stays in cache.
    for (int yy = 0; yy < x.h; yy++) {
        int16_t* drow = xp + ((size_t)(yy + ph) * Wp + pw) * I;
        for (int i = 0; i < I; i++) {
            const float* src = in + (size_t)i * plane + yy * x.w;
            int16_t* dst     = drow + i;
            for (int xx = 0; xx < x.w; xx++) {
                const float v       = src[xx] * inv;
                dst[(size_t)xx * I] = (int16_t)(v + (v >= 0 ? 0.5f : -0.5f));
            }
        }
    }
    t1 = micros(); t_g += t1 - t0; t0 = t1;

    for (int o0 = 0; o0 < O; o0 += B) {
        const int nb = min(B, O - o0);
        for (int b = 0; b < nb; b++) {
            const int8_t* src = q + (size_t)(o0 + b) * K;
            int16_t* dst      = wq + (size_t)b * K;
            for (int i = 0; i < I; i++)
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++) dst[(ky * KW + kx) * I + i] = src[(i * KH + ky) * KW + kx];
        }
        t1 = micros(); t_w += t1 - t0; t0 = t1;
        // Pixels outer, channels inner: one pixel's KH input rows stay in cache
        // while every channel in the block uses them.
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                const int16_t* rows = xp + ((size_t)yy * Wp + xx) * I;
                const int p         = yy * W + xx;
                for (int b = 0; b < nb; b++) {
                    const int16_t* wb = wq + (size_t)b * K;
                    int64_t total     = 0;
                    for (int ky = 0; ky < KH; ky++) total += dot_s16(wb + ky * row_len, rows + (size_t)ky * Wp * I, row_len);
                    out[(size_t)(o0 + b) * HW + p] = bias[o0 + b] + (float)total * (x_scale * scale[o0 + b]);
                }
            }
        }
        t1 = micros(); t_d += t1 - t0; t0 = t1;
    }
    heap_caps_free(wq);
    heap_caps_free(xp);
    if (FACE24_PROFILE) Serial.printf("  conv %3dx%3d k%dx%d %2dx%2d B=%2d | weights %5lu  quantize %5lu  dot %5lu ms  (%.0f MMAC/s in dot)\n",
                  I, O, KH, KW, H, W, B, t_w / 1000, t_g / 1000, t_d / 1000, (double)O * HW * K / (t_d ? t_d : 1));
    return y;
}

Tensor convTranspose(const Tensor& x, const OpDesc& op)  // stride 1, no padding
{
    const TensorDesc& wd = g_tensors[op.w];
    const TensorDesc& bd = g_tensors[op.b];
    const int I = wd.shape[0], O = wd.shape[1], KH = wd.shape[2], KW = wd.shape[3];
    const int H = x.h + KH - 1, W = x.w + KW - 1;
    Tensor y    = makeTensor(O, H, W);
    if (!y.buf || I != x.c || H * W > IMAGE_H * IMAGE_W) return Tensor();

    const int8_t* q    = at<int8_t>(wd.offset);
    const float* scale = at<float>(wd.scale_offset);  // per output channel (axis 1)
    const float* bias  = at<float>(bd.offset);
    const int plane = x.h * x.w, HW = H * W;

    for (int o = 0; o < O; o++) {
        float* acc = g_acc;
        for (int n = 0; n < HW; n++) acc[n] = bias[o];
        const float s = scale[o];
        for (int i = 0; i < I; i++) {
            const float* src = x.data() + (size_t)i * plane;
            for (int ky = 0; ky < KH; ky++) {
                for (int kx = 0; kx < KW; kx++) {
                    const float k = q[(((size_t)i * O + o) * KH + ky) * KW + kx] * s;
                    for (int yy = 0; yy < x.h; yy++) {
                        float* op_      = acc + (yy + ky) * W + kx;
                        const float* ip = src + yy * x.w;
                        for (int j = 0; j < x.w; j++) op_[j] += k * ip[j];
                    }
                }
            }
        }
        memcpy(y.data() + (size_t)o * HW, acc, HW * sizeof(float));
    }
    return y;
}

// PyTorch bilinear, align_corners=False.
Tensor upsample(const Tensor& x, int H, int W)
{
    Tensor y = makeTensor(x.c, H, W);
    if (!y.buf) return y;
    std::vector<int> y0(H), y1(H), x0(W), x1(W);
    std::vector<float> fy(H), fx(W);
    const float sy = (float)x.h / H, sx = (float)x.w / W;
    for (int r = 0; r < H; r++) {
        float s = fmaxf((r + 0.5f) * sy - 0.5f, 0.0f);
        y0[r]   = (int)floorf(s);
        y1[r]   = min(y0[r] + 1, x.h - 1);
        fy[r]   = s - y0[r];
    }
    for (int c = 0; c < W; c++) {
        float s = fmaxf((c + 0.5f) * sx - 0.5f, 0.0f);
        x0[c]   = (int)floorf(s);
        x1[c]   = min(x0[c] + 1, x.w - 1);
        fx[c]   = s - x0[c];
    }
    for (int ch = 0; ch < x.c; ch++) {
        const float* d = x.data() + (size_t)ch * x.h * x.w;
        float* out     = y.data() + (size_t)ch * H * W;
        for (int r = 0; r < H; r++) {
            const float* a = d + y0[r] * x.w;
            const float* b = d + y1[r] * x.w;
            for (int c = 0; c < W; c++) {
                float top    = a[x0[c]] * (1 - fx[c]) + a[x1[c]] * fx[c];
                float bottom = b[x0[c]] * (1 - fx[c]) + b[x1[c]] * fx[c];
                out[r * W + c] = top * (1 - fy[r]) + bottom * fy[r];
            }
        }
    }
    return y;
}

// Models come either from a memory-mapped flash partition (Core2: ESP32 can map ~4 MB of data,
// so one model at a time) or from a file read into PSRAM (T-Deck: models.bin on the SD card).
enum Source { SRC_NONE, SRC_PARTITION, SRC_FILE };
Source g_source        = SRC_NONE;
std::string g_path;
uint8_t* g_file_buf    = nullptr;

bool useModel(const uint8_t* base, int index)
{
    g_base = base;
    g_hdr  = (const Header*)g_base;
    if (memcmp(g_hdr->magic, "F24D", 4) != 0 || g_hdr->version != 1 ||
        g_hdr->payload_bytes != g_entries[index].payload_bytes) {
        g_error = "model data is corrupt or from another version";
        return false;
    }
    g_tensors = (const TensorDesc*)(g_base + sizeof(Header));
    g_ops     = (const OpDesc*)(g_base + sizeof(Header) + g_hdr->n_tensors * sizeof(TensorDesc));
    g_data    = g_base + g_hdr->data_offset;
    g_mapped  = index;
    return true;
}

bool mapModel(int index)
{
    if (g_mapped == index) return true;
    if (g_source == SRC_PARTITION) {
        if (g_mapped >= 0) spi_flash_munmap(g_handle);
        g_mapped        = -1;
        const void* ptr = nullptr;
        esp_err_t err   = esp_partition_mmap(g_part, g_entries[index].offset, g_entries[index].size,
                                             SPI_FLASH_MMAP_DATA, &ptr, &g_handle);
        if (err != ESP_OK) {
            g_error = "could not memory-map model (not enough free flash map pages?)";
            return false;
        }
        if (!useModel((const uint8_t*)ptr, index)) {
            spi_flash_munmap(g_handle);
            return false;
        }
        return true;
    }
    if (g_source == SRC_FILE) {
        // Only one model fits next to the decoder's working memory, so replace the previous one.
        heap_caps_free(g_file_buf);
        g_file_buf = nullptr;
        g_mapped   = -1;
        g_file_buf = (uint8_t*)heap_caps_malloc(g_entries[index].size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!g_file_buf) {
            g_error = "not enough PSRAM for the model";
            return false;
        }
        FILE* f   = fopen(g_path.c_str(), "rb");
        bool ok   = f && fseek(f, g_entries[index].offset, SEEK_SET) == 0 &&
                  fread(g_file_buf, 1, g_entries[index].size, f) == g_entries[index].size;
        if (f) fclose(f);
        if (!ok) {
            heap_caps_free(g_file_buf);
            g_file_buf = nullptr;
            g_error    = "could not read the model from the SD card";
            return false;
        }
        if (!useModel(g_file_buf, index)) {
            heap_caps_free(g_file_buf);
            g_file_buf = nullptr;
            return false;
        }
        return true;
    }
    g_error = "face24::begin...() was not called";
    return false;
}

}  // namespace

bool beginPartition()
{
    if (FACE24_PROFILE) Serial.printf("internal heap free %u, largest block %u\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    g_part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "models");
    if (!g_part) {
        g_error = "no 'models' partition (wrong partition table?)";
        return false;
    }
    uint32_t head[2];
    if (esp_partition_read(g_part, 0, head, sizeof(head)) != ESP_OK || memcmp(head, "F24P", 4) != 0 ||
        head[1] == 0 || head[1] > 8) {
        g_error = "models partition is empty: flash data/models.bin";
        return false;
    }
    g_entries.resize(head[1]);
    if (esp_partition_read(g_part, 8, g_entries.data(), head[1] * sizeof(Entry)) != ESP_OK) {
        g_error = "could not read model directory";
        return false;
    }
    g_source = SRC_PARTITION;
    return true;
}

bool beginFile(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        g_error = "models file not found on the SD card";
        return false;
    }
    uint32_t head[2];
    bool ok = fread(head, 1, sizeof(head), f) == sizeof(head) && memcmp(head, "F24P", 4) == 0 && head[1] > 0 &&
              head[1] <= 8;
    if (ok) {
        g_entries.resize(head[1]);
        ok = fread(g_entries.data(), 1, head[1] * sizeof(Entry), f) == head[1] * sizeof(Entry);
    }
    fclose(f);
    if (!ok) {
        g_error = "models file is not a face24 models.bin";
        return false;
    }
    g_path   = path;
    g_source = SRC_FILE;
    return true;
}

bool preload(size_t len)
{
    for (size_t i = 0; i < g_entries.size(); i++)
        if (g_entries[i].payload_bytes == len) return mapModel(i);
    g_error = "no model for this payload length";
    return false;
}

bool hasModel(size_t len)
{
    for (auto& e : g_entries)
        if (e.payload_bytes == len) return true;
    return false;
}

bool decode(const uint8_t* payload, size_t len, float* out)
{
    g_stats        = Stats();
    uint32_t start = millis();

    int index = -1;
    for (size_t i = 0; i < g_entries.size(); i++)
        if (g_entries[i].payload_bytes == len) index = i;
    if (index < 0) {
        g_error = "no model for this payload length";
        return false;
    }
    uint32_t t = millis();
    if (!mapModel(index)) return false;
    g_stats.map_ms = millis() - t;

    // Codebook lookup: CHW features, 64 x grid_h x grid_w.
    t                 = millis();
    const int gh = g_hdr->grid_h, gw = g_hdr->grid_w, cells = gh * gw;
    const TensorDesc& cb = g_tensors[g_hdr->codebook_a];
    const int dim        = cb.shape[1];
    Tensor x             = makeTensor(dim, gh, gw);
    if (!x.buf) {
        g_error = "out of memory";
        return false;
    }
    const float* cba = at<float>(cb.offset);
    for (int p = 0; p < cells; p++)
        for (int d = 0; d < dim; d++) x.data()[d * cells + p] = cba[payload[p] * dim + d];
    if (g_hdr->kind == 1) {  // residual VQ: second half of the payload indexes codebook 2
        const float* cbb = at<float>(g_tensors[g_hdr->codebook_b].offset);
        for (int p = 0; p < cells; p++)
            for (int d = 0; d < dim; d++) x.data()[d * cells + p] += cbb[payload[cells + p] * dim + d];
    }
    g_stats.other_ms += millis() - t;

    std::vector<Tensor> stack;
    for (uint32_t i = 0; i < g_hdr->n_ops; i++) {
        const OpDesc& op = g_ops[i];
        t                = millis();
        switch (op.type) {
            case OP_CONV:
                x = (op.sh == 1 && op.sw == 1) ? convDot(x, op) : conv(x, op);
                g_stats.conv_ms += millis() - t;
                break;
            case OP_CONVT:
                x = convTranspose(x, op);
                g_stats.convt_ms += millis() - t;
                break;
            case OP_UP2:
                x = upsample(x, x.h * 2, x.w * 2);
                g_stats.up_ms += millis() - t;
                break;
            case OP_UPSIZE:
                x = upsample(x, op.size >> 16, op.size & 0xFFFF);
                g_stats.up_ms += millis() - t;
                break;
            case OP_RELU: {
                x = writable(x);
                float* d = x.data();
                for (size_t n = 0, N = x.size(); n < N; n++) d[n] = d[n] > 0 ? d[n] : 0;
                g_stats.other_ms += millis() - t;
                break;
            }
            case OP_SIGMOID: {
                x = writable(x);
                float* d = x.data();
                for (size_t n = 0, N = x.size(); n < N; n++) d[n] = 1.0f / (1.0f + expf(-d[n]));
                g_stats.other_ms += millis() - t;
                break;
            }
            case OP_RES_BEGIN:
                stack.push_back(x);
                break;
            case OP_RES_END: {
                if (stack.empty()) break;
                Tensor skip = stack.back();
                stack.pop_back();
                x        = writable(x);
                float* d = x.data();
                const float* s = skip.data();
                for (size_t n = 0, N = x.size(); n < N; n++) d[n] += s[n];
                g_stats.other_ms += millis() - t;
                break;
            }
            default:
                g_error = "unknown op in model";
                return false;
        }
        if (!x.buf) {
            g_error = "out of memory or bad layer shape";
            return false;
        }
    }
    if (x.c != 1 || x.h != IMAGE_H || x.w != IMAGE_W) {
        g_error = "decoder produced the wrong image size";
        return false;
    }
    memcpy(out, x.data(), IMAGE_H * IMAGE_W * sizeof(float));
    g_stats.total_ms = millis() - start;
    return true;
}

const char* lastError()
{
    return g_error;
}

const Stats& lastStats()
{
    return g_stats;
}

}  // namespace face24
