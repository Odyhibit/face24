/*
 * face24-core2: scan a face24 QR code with Module13.2 QRCode and show the face.
 *
 * Boots straight to "ready to scan". The scanner uses motion sensing: its light
 * and aimer stay off until something moves in front of it. Payload length picks
 * the codec (64 / 128 / 256 bytes); the face is decoded and drawn at 3x scale.
 *
 * Build with -DFACE24_SELF_TEST=1 to decode the PyTorch test vectors at boot
 * (checks correctness and shows decode times).
 *
 * Hardware: M5Stack Core2 + Module13.2 QRCode, jumpers #3 (RX=G13, TX=G14).
 */
#include <Arduino.h>
#include <M5Unified.h>
#include <M5ModuleQRCode.h>
#include <math.h>
#include <vector>

#include "face24.h"
#include "face24_link.h"

#ifndef FACE24_SELF_TEST
#define FACE24_SELF_TEST 0
#endif
#if FACE24_SELF_TEST
#include "testvectors.h"
#endif

static constexpr int QR_PIN_RX         = 13;
static constexpr int QR_PIN_TX         = 14;
static constexpr size_t RX_BUFFER_SIZE = 2048;
static constexpr size_t MAX_CAPTURE    = 1024;
static constexpr uint32_t FRAME_GAP_MS = 60;

static constexpr int SCALE  = 3;  // 64x80 face -> 192x240, the full screen height
static constexpr int FACE_W = face24::IMAGE_W * SCALE;
static constexpr int PANEL_X = FACE_W + 8;

M5ModuleQRCode module_qrcode;
static float g_image[face24::IMAGE_H * face24::IMAGE_W];
static std::vector<uint8_t> g_capture;
static std::vector<uint8_t> g_shown;  // payload currently on screen
static uint32_t g_last_byte_ms = 0;
static bool g_qr_ok            = false;
static bool g_link_ok          = false;  // ESP-NOW up: scans are also sent to the T-Deck

/* -------------------------------------------------------------- display */

static void drawFace(const float* img)
{
    M5.Display.startWrite();
    for (int y = 0; y < face24::IMAGE_H; y++) {
        for (int x = 0; x < face24::IMAGE_W; x++) {
            float v   = img[y * face24::IMAGE_W + x];
            uint8_t g = (uint8_t)lroundf(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f);
            M5.Display.fillRect(x * SCALE, y * SCALE, SCALE, SCALE, M5.Display.color565(g, g, g));
        }
    }
    M5.Display.endWrite();
}

static void clearPanel()
{
    M5.Display.fillRect(FACE_W, 0, M5.Display.width() - FACE_W, M5.Display.height(), TFT_BLACK);
    M5.Display.setCursor(PANEL_X, 4);
}

static void panelLine(uint16_t color, const char* fmt, ...)
{
    char buf[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    M5.Display.setTextColor(color, TFT_BLACK);
    M5.Display.setCursor(PANEL_X, M5.Display.getCursorY());
    M5.Display.println(buf);
    Serial.println(buf);
}

static void printStats()
{
    const auto& s = face24::lastStats();
    panelLine(TFT_WHITE, "%.2f s", s.total_ms / 1000.0f);
    panelLine(TFT_DARKGREY, "conv  %5lu ms", (unsigned long)s.conv_ms);
    panelLine(TFT_DARKGREY, "convT %5lu ms", (unsigned long)s.convt_ms);
    panelLine(TFT_DARKGREY, "up    %5lu ms", (unsigned long)s.up_ms);
    panelLine(TFT_DARKGREY, "other %5lu ms", (unsigned long)s.other_ms);
    if (s.map_ms) panelLine(TFT_DARKGREY, "map   %5lu ms", (unsigned long)s.map_ms);
}

/* ------------------------------------------------------------ self-test */
#if FACE24_SELF_TEST

static void selfTest()
{
    for (const auto& tv : TEST_VECTORS) {
        if (tv.payload == tv64_0_payload || tv.payload == tv128_0_payload || tv.payload == tv256_0_payload)
            continue;  // one vector per codec keeps boot short
        clearPanel();
        panelLine(TFT_CYAN, "self-test %d B", tv.size);
        panelLine(TFT_WHITE, "decoding...");
        if (!face24::decode(tv.payload, tv.size, g_image)) {
            panelLine(TFT_RED, "%s", face24::lastError());
            delay(3000);
            continue;
        }
        float worst = 0;
        for (int i = 0; i < face24::IMAGE_H * face24::IMAGE_W; i++) worst = fmaxf(worst, fabsf(g_image[i] - tv.recon[i]));
        drawFace(g_image);
        clearPanel();
        panelLine(TFT_CYAN, "self-test %d B", tv.size);
        printStats();
        bool ok = worst * 255.0f < 0.5f;
        panelLine(ok ? TFT_GREEN : TFT_RED, "%s diff %.3f", ok ? "PASS" : "FAIL", worst * 255.0f);
        delay(1500);
    }
}
#endif

/* ------------------------------------------------------------------ QR */

static bool initQr()
{
    auto cfg     = module_qrcode.getConfig();
    cfg.pin_rx   = QR_PIN_RX;
    cfg.pin_tx   = QR_PIN_TX;
    cfg.baudrate = 115200;
    cfg.serial   = &Serial1;
    module_qrcode.setConfig(cfg);
    Serial1.setRxBufferSize(RX_BUFFER_SIZE);
    if (!module_qrcode.begin()) return false;

    std::string fw;
    for (int i = 0; i < 10 && fw.empty(); i++) {
        delay(300);
        fw = module_qrcode.getFirmwareVersion();
    }
    // Light and aimer stay off until the module sees motion, then light up while
    // decoding (same settings as the library's MotionSensingMode example).
    module_qrcode.setTriggerMode(QRCodeM14::TRIGGER_MODE_MOTION_SENSING);
    module_qrcode.setMotionSensitivity(1);
    module_qrcode.setFillLightMode(QRCodeM14::FILL_LIGHT_ON_DECODE);
    module_qrcode.setPosLightMode(QRCodeM14::POS_LIGHT_FLASH_ON_DECODE);
    delay(50);
    while (Serial1.available()) Serial1.read();
    return !fw.empty();
}

static void handleScan(const std::vector<uint8_t>& payload)
{
    if (payload == g_shown) {
        // Already decoded here: don't decode again, but do send it again, in case the T-Deck
        // missed it (e.g. it was off, or still starting up).
        if (g_link_ok) {
            bool sent = face24link::send(payload.data(), payload.size());
            clearPanel();
            panelLine(TFT_CYAN, "face24 %u B", (unsigned)payload.size());
            panelLine(sent ? TFT_GREEN : TFT_RED, sent ? "sent to T-Deck again" : "send failed");
            printStats();
        }
        return;
    }

    if (!face24::hasModel(payload.size())) {
        clearPanel();
        panelLine(TFT_ORANGE, "not a face code");
        panelLine(TFT_DARKGREY, "(%u bytes)", (unsigned)payload.size());
        return;
    }
    // Send first, so the T-Deck starts decoding while this device decodes too.
    bool sent = g_link_ok && face24link::send(payload.data(), payload.size());
    clearPanel();
    panelLine(TFT_CYAN, "face24 %u B", (unsigned)payload.size());
    if (g_link_ok) panelLine(sent ? TFT_GREEN : TFT_RED, sent ? "sent to T-Deck" : "send failed");
    panelLine(TFT_WHITE, "decoding...");
    bool ok = face24::decode(payload.data(), payload.size(), g_image);

    if (ok) {
        drawFace(g_image);
        g_shown = payload;
        clearPanel();
        panelLine(TFT_CYAN, "face24 %u B", (unsigned)payload.size());
        if (g_link_ok) panelLine(sent ? TFT_GREEN : TFT_RED, sent ? "sent to T-Deck" : "send failed");
        printStats();
    } else {
        clearPanel();
        panelLine(TFT_RED, "decode failed:");
        panelLine(TFT_RED, "%s", face24::lastError());
    }
    // Scans that queued up during the decode are stale.
    while (Serial1.available()) Serial1.read();
    g_capture.clear();
}

/* --------------------------------------------------------------- main */

void setup()
{
    Serial.begin(115200);
    M5.begin();
    M5.Display.setFont(&fonts::Font2);
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setCursor(PANEL_X, 4);
    Serial.println("\nface24-core2");

    if (!face24::beginPartition()) {
        M5.Display.setCursor(0, 0);
        M5.Display.setTextColor(TFT_RED);
        M5.Display.println(face24::lastError());
        Serial.println(face24::lastError());
        while (true) delay(1000);
    }

#if FACE24_SELF_TEST
    selfTest();
#endif

    M5.Display.setTextDatum(middle_center);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.drawString("Starting scanner...", M5.Display.width() / 2, M5.Display.height() / 2);
    g_qr_ok   = initQr();
    g_link_ok = face24link::beginSender();
    Serial.printf("ESP-NOW sender: %s\n", g_link_ok ? "ready" : "failed");
    M5.Display.fillScreen(TFT_BLACK);
    if (g_qr_ok) {
        M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
        M5.Display.drawString("Scan a face QR code", M5.Display.width() / 2, M5.Display.height() / 2);
    } else {
        M5.Display.setTextColor(TFT_RED, TFT_BLACK);
        M5.Display.drawString("QR module not responding", M5.Display.width() / 2, M5.Display.height() / 2);
    }
    M5.Display.setTextDatum(top_left);
    g_capture.reserve(MAX_CAPTURE);
}

void loop()
{
    M5.update();
    while (Serial1.available()) {
        int c = Serial1.read();
        if (g_capture.size() < MAX_CAPTURE) g_capture.push_back((uint8_t)c);
        g_last_byte_ms = millis();
    }
    if (!g_capture.empty() && millis() - g_last_byte_ms > FRAME_GAP_MS) {
        std::vector<uint8_t> payload;
        payload.swap(g_capture);
        handleScan(payload);
    }
}
