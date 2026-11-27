# face24-tdeck

This is the display end of the face24 demo. The Core2 scans a face24 QR code and broadcasts the
payload over ESP-NOW. This firmware, running on a LilyGo T-Deck Plus, receives the payload,
decodes the face and shows it.

- **Link:** ESP-NOW broadcast on Wi-Fi channel 1. There is no pairing or access point. See
  `../face24-lib/src/face24_link.h`.
- **Decoder:** the same engine as the Core2 (`../face24-lib`). It has int8 weights and int16
  convolutions on the MAC16 unit, which the ESP32-S3 also has.
- **Models:** read from the SD card. The decoder for the incoming payload length is loaded into
  PSRAM. The first face of each size takes longer while its model loads. After that the model
  stays loaded until a different size arrives.

## SD card

Put these on the card. It must be FAT32, which the Launcher already needs.

| File on the SD card | From |
|---|---|
| `/face24/models.bin` | `../face24-core2/data/models.bin` (10.7 MB), built by `face24-core2/tools/make_device_models.py` |
| `face24-tdeck.bin` (anywhere the Launcher can browse) | `face24-tdeck.bin`, or `.pio/build/tdeck/firmware.bin` after `pio run` |

Install `face24-tdeck.bin` from the Launcher's SD-card menu. To flash it over USB instead, run
`pio run -t upload`.

## Use

1. Start the T-Deck. It shows **Waiting for a face**.
2. Scan a face QR code on the Core2. The Core2 shows "sent to T-Deck" and decodes the face too.
3. The T-Deck shows "decoding...", then the face and the decode time.

## If something looks wrong

- **The screen looks like a photo negative** (white background at startup): build with
  `-DTDECK_INVERT=0` in `build_flags`.
- **"No SD card" or "models file not found":** check the card and the path `/face24/models.bin`.
- **Nothing arrives:** both devices must run their face24 firmware. They use channel 1 and
  don't need Wi-Fi credentials. The Core2's panel says "send failed" if ESP-NOW couldn't send.
