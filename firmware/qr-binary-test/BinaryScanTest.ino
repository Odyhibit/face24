/*
 * BinaryScanTest
 *
 * Checks whether the QR module passes raw binary (QR byte-mode) data through
 * to the UART unchanged, or whether its firmware transcodes / filters bytes.
 *
 * 1. Generate the test codes:  python3 make_test_qr.py   (needs: pip install segno)
 * 2. Flash this sketch and scan seq_256.png first, then the rand_*.png codes.
 * 3. The screen shows PASS/FAIL; a full hex dump goes to USB Serial (115200).
 *
 * This sketch only READS the module's persistent settings (packing, prefix,
 * suffix, case conversion, encoding) and reports them. The only setting it
 * writes is trigger mode = auto, the same as the AutoMode example.
 *
 * Framing: the bytes of one scan are collected until the UART has been quiet
 * for FRAME_GAP_MS, so no packet format is needed for this test.
 */
#include <Arduino.h>
#include <M5Unified.h>
#include <M5ModuleQRCode.h>
#include <vector>

M5ModuleQRCode module_qrcode;

static constexpr size_t RX_BUFFER_SIZE = 4096;
static constexpr size_t MAX_CAPTURE    = 4096;
static constexpr uint32_t FRAME_GAP_MS = 60;

struct KnownPayload {
    const char* name;
    size_t len;
    uint32_t crc32;
};
// Must match make_test_qr.py output.
static const KnownPayload KNOWN[] = {
    {"seq_256", 256, 0x29058C73},
    {"rand_64", 64, 0xC6EBC1E3},
    {"rand_128", 128, 0xF39396AF},
    {"rand_256", 256, 0xFDAEFC32},
};

static std::vector<uint8_t> capture;
static uint32_t last_byte_ms = 0;
static bool overflowed       = false;
static uint32_t scan_count   = 0;
static uint32_t pass_count   = 0;

/* -------------------------------------------------------------------------- */

static uint32_t crc32(const uint8_t* data, size_t len)  // zlib-compatible
{
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xEDB88320 & -(int32_t)(crc & 1));
    }
    return ~crc;
}

static void drainUart()
{
    delay(50);
    while (Serial1.available()) Serial1.read();
}

// Configuration read: 23 PID FID -> 24 PID FID VAL. Returns -1 if no reply.
static int readConfig(uint8_t pid, uint8_t fid)
{
    drainUart();
    const uint8_t cmd[] = {0x23, pid, fid};
    Serial1.write(cmd, sizeof(cmd));

    uint8_t rx[4];
    size_t n       = 0;
    uint32_t start = millis();
    while (millis() - start < 300 && n < sizeof(rx)) {
        if (Serial1.available()) {
            uint8_t c = Serial1.read();
            if (n == 0 && c != 0x24) continue;  // skip until reply header
            rx[n++] = c;
        }
    }
    if (n == 4 && rx[1] == pid && rx[2] == fid) return rx[3];
    return -1;
}

static void hexDump(const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; i += 16) {
        Serial.printf("%04X: ", (unsigned)i);
        for (size_t j = i; j < i + 16 && j < len; j++) Serial.printf("%02X ", data[j]);
        Serial.println();
    }
}

static void report(const char* label, int value, int expect_off)
{
    String s = String(label) + ": " + (value < 0 ? String("no reply") : String(value));
    bool warn = value >= 0 && expect_off >= 0 && value != expect_off;
    M5.Display.setTextColor(warn ? TFT_ORANGE : TFT_WHITE);
    M5.Display.println(s + (warn ? "  <-- not off" : ""));
    Serial.println(s);
}

/* -------------------------------------------------------------------------- */

static void analyze(const std::vector<uint8_t>& raw)
{
    scan_count++;
    Serial.printf("\n===== Scan #%lu: %u raw bytes%s =====\n", (unsigned long)scan_count, (unsigned)raw.size(),
                  overflowed ? " (CAPTURE OVERFLOW)" : "");
    hexDump(raw.data(), raw.size());

    // If data packing happens to be enabled (0x03 + LEN(2) + DATA), strip it.
    const uint8_t* p  = raw.data();
    size_t len        = raw.size();
    const char* frame = "none";
    if (len >= 3 && raw[0] == 0x03) {
        size_t be = (raw[1] << 8) | raw[2];
        size_t le = (raw[2] << 8) | raw[1];
        if (be == len - 3) {
            frame = "0x03+LEN(BE)";
            p += 3;
            len -= 3;
        } else if (le == len - 3) {
            frame = "0x03+LEN(LE)";
            p += 3;
            len -= 3;
        }
    }

    uint32_t crc              = crc32(p, len);
    const KnownPayload* match = nullptr;
    for (auto& k : KNOWN) {
        if (k.len == len && k.crc32 == crc) match = &k;
    }

    // Byte-by-byte comparison against the 0x00..0xFF sequence, if it looks like one.
    bool looks_seq     = len >= 4 && p[0] == 0x00 && p[1] == 0x01 && p[2] == 0x02 && p[3] == 0x03;
    long first_bad     = -1;
    if (looks_seq && !match) {
        for (size_t i = 0; i < len; i++) {
            if (i >= 256 || p[i] != (uint8_t)i) {
                first_bad = i;
                break;
            }
        }
        if (first_bad < 0 && len < 256) first_bad = len;  // truncated
    }

    bool pass = match != nullptr && !overflowed;
    if (pass) pass_count++;

    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.printf("Scan #%lu   (passed %lu)\n", (unsigned long)scan_count, (unsigned long)pass_count);
    M5.Display.printf("raw %u B, payload %u B\n", (unsigned)raw.size(), (unsigned)len);
    M5.Display.printf("framing: %s\n", frame);
    M5.Display.printf("crc32: %08lX\n\n", (unsigned long)crc);

    M5.Display.setTextSize(2);
    M5.Display.setTextColor(pass ? TFT_GREEN : TFT_RED);
    M5.Display.println(pass ? "PASS" : "FAIL");
    M5.Display.setTextSize(1);

    if (match) {
        M5.Display.setTextColor(TFT_GREEN);
        M5.Display.printf("%s matches exactly\n", match->name);
        Serial.printf("PASS: %s matches exactly\n", match->name);
    } else {
        M5.Display.setTextColor(TFT_YELLOW);
        if (overflowed) M5.Display.println("capture buffer overflowed");
        if (looks_seq && first_bad >= 0) {
            M5.Display.printf("seq_256 differs at 0x%02lX\n", first_bad);
            if ((size_t)first_bad < len) {
                M5.Display.printf("expected %02X, got %02X\n", (unsigned)(first_bad & 0xFF), p[first_bad]);
            } else {
                M5.Display.println("data ends early");
            }
            if (first_bad == 0x80 && len > 256) M5.Display.println("high bytes expanded: likely UTF-8");
            if (first_bad == 0x80 && len == 256 && p[0x80] == '?') M5.Display.println("high bytes replaced by '?'");
        } else if (len > 0 && p[0] != 0x00) {
            M5.Display.println("unknown payload (not a test code,");
            M5.Display.println("or leading 0x00 was dropped)");
        }
        if (len >= 2 && p[len - 2] == '\r' && p[len - 1] == '\n') M5.Display.println("ends in CR LF: suffix on?");
        Serial.printf("FAIL: len=%u crc32=%08lX first_bad=%ld\n", (unsigned)len, (unsigned long)crc, first_bad);
    }
}

/* -------------------------------------------------------------------------- */

void setup()
{
    Serial.begin(115200);
    M5.begin();
    M5.Display.setFont(&fonts::Font2);
    M5.Display.setTextScroll(false);
    Serial.println("\nBinaryScanTest");

    // Core2 + module jumpers #3 route the QR UART to G13/G14 (Core2 Port C UART).
    // Both directions are tried; the one the module answers on is kept.
    // Override with -DQR_PIN_RX=.. -DQR_PIN_TX=.. to skip the probe.
#if defined(QR_PIN_RX) && defined(QR_PIN_TX)
    const int candidates[][2] = {{QR_PIN_RX, QR_PIN_TX}};
#else
    const int candidates[][2] = {{13, 14}, {14, 13}};  // {rx, tx}
#endif
    const size_t n_candidates = sizeof(candidates) / sizeof(candidates[0]);

    auto cfg     = module_qrcode.getConfig();
    cfg.pin_rx   = candidates[0][0];
    cfg.pin_tx   = candidates[0][1];
    cfg.baudrate = 115200;
    cfg.serial   = &Serial1;
    module_qrcode.setConfig(cfg);

    // Must happen before Serial1.begin() (called inside module_qrcode.begin()).
    Serial1.setRxBufferSize(RX_BUFFER_SIZE);

    M5.Display.println("BinaryScanTest");
    while (!module_qrcode.begin()) {
        M5.Display.setTextColor(TFT_RED);
        M5.Display.println("Init module failed, retry...");
        delay(1000);
    }

    std::string fw;
    int pin_rx = candidates[0][0], pin_tx = candidates[0][1];
    for (size_t c = 0; c < n_candidates && fw.empty(); c++) {
        int rx = candidates[c][0], tx = candidates[c][1];
        M5.Display.setTextColor(TFT_WHITE);
        M5.Display.printf("trying RX=G%d TX=G%d ... ", rx, tx);
        if (c > 0) {
            Serial1.end();
            Serial1.setRxBufferSize(RX_BUFFER_SIZE);
            Serial1.begin(115200, SERIAL_8N1, rx, tx);
        }
        // The module may need a moment after power-on before it answers.
        for (int i = 0; i < (c == 0 ? 8 : 4) && fw.empty(); i++) {
            delay(300);
            fw = module_qrcode.getFirmwareVersion();
        }
        M5.Display.setTextColor(fw.empty() ? TFT_RED : TFT_GREEN);
        M5.Display.println(fw.empty() ? "no reply" : "OK");
        Serial.printf("RX=G%d TX=G%d: %s\n", rx, tx, fw.empty() ? "no reply" : fw.c_str());
        if (!fw.empty()) {
            pin_rx = rx;
            pin_tx = tx;
        }
    }
    if (fw.empty() && n_candidates > 1) {
        // Nothing answered: fall back to the first pair and still listen for scans.
        Serial1.end();
        Serial1.setRxBufferSize(RX_BUFFER_SIZE);
        Serial1.begin(115200, SERIAL_8N1, pin_rx, pin_tx);
    }
    delay(1500);  // leave the result lines readable before the settings list

    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.printf("QR UART RX=G%d TX=G%d\n", pin_rx, pin_tx);

    module_qrcode.setTriggerMode(QRCodeM14::TRIGGER_MODE_AUTO);

    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.println("Module settings:");
    M5.Display.printf("fw: %s\n", fw.empty() ? "(no reply)" : fw.c_str());
    report("data packing (5143)", readConfig(0x51, 0x43), 0);
    report("suffix enable (514C)", readConfig(0x51, 0x4C), 0);
    report("prefix enable (51CB)", readConfig(0x51, 0xCB), 0);
    report("case convert (5148)", readConfig(0x51, 0x48), 0);
    report("kbd encoding (4341)", readConfig(0x43, 0x41), -1);
    report("trigger mode (6141)", readConfig(0x61, 0x41), -1);

    drainUart();
    capture.reserve(MAX_CAPTURE);
    M5.Display.setTextColor(TFT_CYAN);
    M5.Display.println("\nScan seq_256.png first.");
}

void loop()
{
    M5.update();

    while (Serial1.available()) {
        int c = Serial1.read();
        if (capture.size() < MAX_CAPTURE) {
            capture.push_back((uint8_t)c);
        } else {
            overflowed = true;
        }
        last_byte_ms = millis();
    }

    if (!capture.empty() && millis() - last_byte_ms > FRAME_GAP_MS) {
        analyze(capture);
        capture.clear();
        overflowed = false;
    }
}
