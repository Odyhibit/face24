// face24 decoder for ESP32 / ESP32-S3: int8 weights, int16 activations into each convolution
// (MAC16 dot products), float32 elsewhere. Port of face24's web/face24.js decoder.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace face24 {

static constexpr int IMAGE_H = 80;
static constexpr int IMAGE_W = 64;

struct Stats {
    uint32_t total_ms = 0;
    uint32_t map_ms   = 0;  // mapping or loading the model (only when switching models)
    uint32_t conv_ms  = 0;
    uint32_t convt_ms = 0;
    uint32_t up_ms    = 0;
    uint32_t other_ms = 0;  // relu, sigmoid, residual adds, codebook lookup
};

// Model source; call one of these once.
//   beginPartition(): the "models" flash partition, memory-mapped (Core2).
//   beginFile(path):  models.bin as a file (e.g. "/sd/face24/models.bin"), read into PSRAM (T-Deck).
bool beginPartition();
bool beginFile(const char* path);
inline bool begin()
{
    return beginPartition();
}

// Loads the model for this payload length now (e.g. at boot), so the first decode doesn't wait.
bool preload(size_t len);

// True if a model exists for this payload length (64, 128 or 256 bytes).
bool hasModel(size_t payload_len);

// Decodes one payload into out[IMAGE_H * IMAGE_W], values in [0, 1].
bool decode(const uint8_t* payload, size_t len, float* out);

const char* lastError();
const Stats& lastStats();

}  // namespace face24
