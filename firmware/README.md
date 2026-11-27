# Face24 hardware demo

Scan a face24 QR code with an M5Stack Core2 and see the face, on the Core2 and, a few seconds
later, on a LilyGo T-Deck Plus over ESP-NOW. The devices use no network and no phone.

```
printed / on-screen QR ──► Core2 + QR module ──ESP-NOW──► T-Deck Plus
                           decodes too (~8–10 s)          decodes (~3–3.5 s), 3× face
```

| Folder | What it is |
|---|---|
| `face24-core2/` | Core2 scanner firmware. It reads the QR, sends the payload over ESP-NOW and decodes locally. It also holds the model converter (`tools/make_device_models.py`) |
| `face24-tdeck/` | T-Deck Plus receiver firmware. It decodes with models loaded from the SD card into PSRAM |
| `face24-lib/` | Shared decoder (a C++ port of `web/face24.js`) and the ESP-NOW packet format |
| `qr-binary-test/` | Check that the QR module passes raw binary payloads through unchanged |
| `lib/M5Module-QRCode/` | M5Stack's Module13.2 QRCode library (MIT, unchanged copy) |

## Hardware

- **M5Stack Core2** with **Module13.2 QRCode**: module switch set to **UART**, and **jumpers #3**
  on RX and TX (Core2 G13 / G14). The example sketches' G16/G17 are PSRAM pins on the Core2, and
  G1 is the USB serial line.
- **LilyGo T-Deck Plus** with an SD card (FAT32), optionally running the SD-card Launcher.

## Build and install

All projects are PlatformIO (`espressif32@6.7.0`). M5Unified is pinned to 0.2.17, because 0.3.x
breaks the QR library.

1. **Make the device models** (needs numpy). This writes `face24-core2/data/models.bin` (10.7 MB)
   and the self-test vectors:
   ```bash
   cd face24-core2 && python3 tools/make_device_models.py
   ```
2. **Core2:** `pio run -t upload`, then write the models into their flash partition:
   ```bash
   python3 ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32 --baud 921600 \
       write_flash 0x310000 data/models.bin
   ```
3. **T-Deck:** in `face24-tdeck`, run `pio run`. Copy `.pio/build/tdeck/firmware.bin` and
   `../face24-core2/data/models.bin` to the SD card. The models must go at `/face24/models.bin`.
   Install the firmware from the Launcher. **Don't use `pio run -t upload` on a T-Deck that runs the
   Launcher:** it overwrites the Launcher. Use USB only for the serial monitor.

## How fast

These are for a 64×80 face, with the decoder already loaded.

| | 64 B | 128 B | 256 B |
|---|---|---|---|
| Core2 (ESP32, MAC16 int16 dot products) | 8.4 s | 10.5 s | 10.5 s |
| T-Deck Plus (ESP32-S3, PIE vector dot products) | 3.2 s | ~3.5 s | ~3.5 s |

The T-Deck loads a model from the SD card (about 4.6 s) only when the payload size changes. It
loads the 128-byte model at boot. Both firmwares have a self-test build that checks every codec
against PyTorch's output: on the Core2 use `-DFACE24_SELF_TEST=1`, on the T-Deck use
`pio run -e tdeck-selftest`.

## Findings worth remembering

- The QR module passes raw binary QR payloads through byte for byte, including `0x00` and
  `0x80`–`0xFF`, with no character conversion (`qr-binary-test/`). Its config writes (`0x21 …`)
  are saved in the module's own flash.
- The Core2's ESP32 can memory-map only about 4 MB of flash data at once, so it maps one 3.5 MB
  model at a time. The T-Deck reads the selected model from the SD card into PSRAM instead.
