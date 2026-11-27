# face24-core2

Scan a [face24](https://github.com/Odyhibit/face24) QR code with an M5Stack Core2 and
Module13.2 QRCode, and show the decoded face on the screen.

- The payload length picks the codec: 64, 128 or 256 bytes.
- The decoder is a C++ port of face24's `web/face24.js`. It uses int8 weights with a scale per
  output channel and float32 activations, so it gives the same quality as the web demo.
- It boots straight to "Scan a face QR code". The scanner is in motion-sensing mode: its light
  and aimer stay off until something moves in front of it. These module settings are saved in
  the module's own flash.
- Decoding takes about 8 s for 64-byte payloads and 10.5 s for 128 or 256 bytes. The
  convolutions use int16 activations on the ESP32's MAC16 unit.
- After each scan it also broadcasts the payload over ESP-NOW (Wi-Fi channel 1, no pairing) to the
  T-Deck receiver in `../face24-tdeck`. The panel shows "sent to T-Deck".
- The decoder and the ESP-NOW link live in `../face24-lib`, which both firmwares share.
- Build flags:
  - `-DFACE24_SELF_TEST=1` decodes the PyTorch test vectors at boot and shows PASS/FAIL and
    the decode times.
  - `-DFACE24_PROFILE=1` prints per-layer timings over USB serial.

## Hardware

- M5Stack Core2
- Module13.2 QRCode, with **jumpers #3** on both RX and TX (Core2 RX = G13, TX = G14)
- Module switch set to **UART**

## Build and flash

```bash
# 1. Convert this repo's web models (needs numpy). Writes data/models.bin and src/testvectors.h
python3 tools/make_device_models.py

# 2. Flash the app. This also writes the partition table.
pio run -t upload

# 3. Flash the models into the "models" partition. Only needed when the models change.
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py --chip esp32 --baud 921600 \
    write_flash 0x310000 data/models.bin
```

## Flash layout

| Partition | Offset | Size | Contents |
|---|---|---|---|
| app0 | 0x10000 | 3 MB | firmware |
| models | 0x310000 | 12 MB | `data/models.bin`: three decoders, about 3.5 MB each |

The original ESP32 can map only about 4 MB of flash for data at once, so only one model is
mapped at a time. Switching codecs remaps it, which takes a few milliseconds.
