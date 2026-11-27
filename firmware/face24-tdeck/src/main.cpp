/*
 * face24-tdeck: the display end of the face24 demo.
 *
 * The Core2 scans a face24 QR code and broadcasts the payload over ESP-NOW. This firmware
 * receives it, loads the matching decoder from the SD card (/face24/models.bin) into PSRAM,
 * decodes the face and draws it at 3x scale.
 *
 * Hardware: LilyGo T-Deck Plus (ESP32-S3). Display, SD card and LoRa radio share one SPI bus.
 */
#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <math.h>
#include <vector>

#include "face24.h"
#include "face24_link.h"

#ifndef FACE24_SELF_TEST
#define FACE24_SELF_TEST 0  // 1: decode the PyTorch test vectors at boot (env:tdeck-selftest)
#endif
#if FACE24_SELF_TEST
#include "testvectors.h"  // from ../face24-core2/src (made by tools/make_device_models.py)
#endif

// T-Deck pins (LilyGo T-Deck examples/UnitTest/utilities.h)
static constexpr int BOARD_POWERON = 10;  // peripheral power: must be HIGH
static constexpr int TFT_CS        = 12;
static constexpr int TFT_DC        = 11;
static constexpr int TFT_BL        = 42;
static constexpr int SPI_SCK       = 40;
static constexpr int SPI_MOSI      = 41;
static constexpr int SPI_MISO      = 38;
static constexpr int SD_CS         = 39;
static constexpr int RADIO_CS      = 9;

#ifndef TDECK_INVERT
#define TDECK_INVERT 1  // the T-Deck's ST7789 needs colour inversion; set 0 if the screen looks negative
#endif

static const char* MODELS_PATH = "/sd/face24/models.bin";

class LGFX_TDeck : public lgfx::LGFX_Device {
    lgfx::Panel_ST7789 panel_;
    lgfx::Bus_SPI bus_;

   public:
    LGFX_TDeck()
    {
        auto b        = bus_.config();
        b.spi_host    = SPI2_HOST;
        b.spi_mode    = 0;
        b.freq_write  = 40000000;
        b.freq_read   = 16000000;
        b.spi_3wire   = false;
        b.use_lock    = true;
        b.dma_channel = SPI_DMA_CH_AUTO;
        b.pin_sclk    = SPI_SCK;
        b.pin_mosi    = SPI_MOSI;
        b.pin_miso    = SPI_MISO;
        b.pin_dc      = TFT_DC;
        bus_.config(b);
        panel_.setBus(&bus_);

        auto p         = panel_.config();
        p.pin_cs       = TFT_CS;
        p.pin_rst      = -1;
        p.pin_busy     = -1;
        p.panel_width  = 240;
        p.panel_height = 320;
        p.readable     = false;
        p.invert       = TDECK_INVERT;
        p.rgb_order    = false;
        p.dlen_16bit   = false;
        p.bus_shared   = true;  // the SD card is on the same bus
        panel_.config(p);
        setPanel(&panel_);
    }
};

static LGFX_TDeck tft;
static face24link::Receiver g_rx;
static uint8_t g_payload[face24link::MAX_PAYLOAD];
static std::vector<uint8_t> g_shown;
static float g_image[face24::IMAGE_H * face24::IMAGE_W];
static bool g_models_ok = false;

static constexpr int SCALE   = 3;  // 64x80 face -> 192x240, the full screen height
static constexpr int FACE_W  = face24::IMAGE_W * SCALE;
static constexpr int PANEL_X = FACE_W + 8;

/* -------------------------------------------------------------- display */

static void drawFace(const float* img)
{
    tft.startWrite();
    for (int y = 0; y < face24::IMAGE_H; y++) {
        for (int x = 0; x < face24::IMAGE_W; x++) {
            float v   = img[y * face24::IMAGE_W + x];
            uint8_t g = (uint8_t)lroundf(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f);
            tft.fillRect(x * SCALE, y * SCALE, SCALE, SCALE, tft.color565(g, g, g));
        }
    }
    tft.endWrite();
}

static void clearPanel()
{
    tft.fillRect(FACE_W, 0, tft.width() - FACE_W, tft.height(), TFT_BLACK);
    tft.setCursor(PANEL_X, 4);
}

static void panelLine(uint16_t color, const char* fmt, ...)
{
    char buf[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    tft.setTextColor(color, TFT_BLACK);
    tft.setCursor(PANEL_X, tft.getCursorY());
    tft.println(buf);
    Serial.println(buf);
}

static void centerMessage(uint16_t color, const char* line1, const char* line2 = nullptr)
{
    tft.fillScreen(TFT_BLACK);
    tft.setTextDatum(middle_center);
    tft.setTextColor(color, TFT_BLACK);
    tft.drawString(line1, tft.width() / 2, tft.height() / 2 - (line2 ? 10 : 0));
    if (line2) tft.drawString(line2, tft.width() / 2, tft.height() / 2 + 12);
    tft.setTextDatum(top_left);
    Serial.printf("%s %s\n", line1, line2 ? line2 : "");
}

/* ------------------------------------------------------------- payloads */

static void onReceive(const uint8_t* mac, const uint8_t* data, int len)
{
    g_rx.onPacket(data, len);  // Wi-Fi task: just reassemble; decoding happens in loop()
}

static void handlePayload(const uint8_t* payload, size_t len)
{
    if (g_shown.size() == len && memcmp(g_shown.data(), payload, len) == 0) return;  // already on screen
    if (!face24::hasModel(len)) {
        clearPanel();
        panelLine(TFT_ORANGE, "not a face code");
        panelLine(TFT_DARKGREY, "(%u bytes)", (unsigned)len);
        return;
    }
    clearPanel();
    panelLine(TFT_CYAN, "face24 %u B", (unsigned)len);
    panelLine(TFT_WHITE, "decoding...");
    if (!face24::decode(payload, len, g_image)) {
        clearPanel();
        panelLine(TFT_RED, "decode failed:");
        panelLine(TFT_RED, "%s", face24::lastError());
        return;
    }
    drawFace(g_image);
    g_shown.assign(payload, payload + len);
    const auto& s = face24::lastStats();
    clearPanel();
    panelLine(TFT_CYAN, "face24 %u B", (unsigned)len);
    panelLine(TFT_WHITE, "%.2f s", s.total_ms / 1000.0f);
    if (s.map_ms) panelLine(TFT_DARKGREY, "model load %lu ms", (unsigned long)s.map_ms);
    panelLine(TFT_DARKGREY, "conv  %5lu ms", (unsigned long)s.conv_ms);
    panelLine(TFT_DARKGREY, "other %5lu ms", (unsigned long)(s.convt_ms + s.up_ms + s.other_ms));
}

/* ------------------------------------------------------------ self-test */
#if FACE24_SELF_TEST
static void selfTest()
{
    for (const auto& tv : TEST_VECTORS) {
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
        const auto& s = face24::lastStats();
        drawFace(g_image);
        clearPanel();
        bool ok = worst * 255.0f < 0.5f;
        panelLine(TFT_CYAN, "self-test %d B", tv.size);
        panelLine(TFT_WHITE, "%.2f s", s.total_ms / 1000.0f);
        if (s.map_ms) panelLine(TFT_DARKGREY, "model load %lu ms", (unsigned long)s.map_ms);
        panelLine(TFT_DARKGREY, "conv  %5lu ms", (unsigned long)s.conv_ms);
        panelLine(TFT_DARKGREY, "other %5lu ms", (unsigned long)(s.convt_ms + s.up_ms + s.other_ms));
        panelLine(ok ? TFT_GREEN : TFT_RED, "%s diff %.3f", ok ? "PASS" : "FAIL", worst * 255.0f);
        delay(1500);
    }
}
#endif

/* ----------------------------------------------------------------- main */

void setup()
{
    Serial.begin(115200);

    pinMode(BOARD_POWERON, OUTPUT);
    digitalWrite(BOARD_POWERON, HIGH);
    // Everything on the shared SPI bus deselected before any of it is used.
    for (int cs : {TFT_CS, SD_CS, RADIO_CS}) {
        pinMode(cs, OUTPUT);
        digitalWrite(cs, HIGH);
    }
    delay(50);

    tft.init();
    tft.setRotation(1);
    tft.setFont(&fonts::Font2);
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);
    centerMessage(TFT_WHITE, "face24 receiver", "starting...");

    SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
    // 40 MHz roughly halves model loading; fall back to 20 MHz for cards that can't keep up.
    bool sd = SD.begin(SD_CS, SPI, 40000000);
    if (!sd) {
        SD.end();
        sd = SD.begin(SD_CS, SPI, 20000000);
    }
    Serial.printf("SD card: %s\n", sd ? "ok" : "not found");
    if (!sd) {
        centerMessage(TFT_RED, "No SD card", "insert one with /face24/models.bin");
    } else if (!(g_models_ok = face24::beginFile(MODELS_PATH))) {
        centerMessage(TFT_RED, face24::lastError(), "expected /face24/models.bin");
    } else {
#if FACE24_SELF_TEST
        selfTest();
#endif
        // 128 bytes is the default codec: load it now so the first scan doesn't wait for the SD card.
        centerMessage(TFT_WHITE, "Loading the 128-byte model...");
        uint32_t t = millis();
        if (!face24::preload(128)) centerMessage(TFT_RED, "Model load failed", face24::lastError());
        Serial.printf("128-byte model preloaded in %lu ms\n", (unsigned long)(millis() - t));
    }

    bool link = face24link::begin() && esp_now_register_recv_cb(onReceive) == ESP_OK;
    if (!link) {
        centerMessage(TFT_RED, "ESP-NOW failed to start");
    } else if (g_models_ok) {
        centerMessage(TFT_GREEN, "Waiting for a face", "scan a QR code on the Core2");
    }
}

void loop()
{
    size_t n = g_rx.take(g_payload);
    if (n && g_models_ok) handlePayload(g_payload, n);
    delay(10);
}
