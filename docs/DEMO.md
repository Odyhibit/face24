# Face24 / Codec Lab

A local, four-way comparison of the same aligned 64×80 grayscale face. The demo runs inference only; it does not train models, load ArcFace, change datasets, or send photographs to an external service.

| Option | QR contents | Selected model |
|---|---|---|
| 64 bytes | 64 raw codebook indices | Phase 18, epoch 147 |
| 128 bytes (default) | 128 raw codebook indices | Phase 21, epoch 187 |
| 256 bytes | 128 coarse + 128 residual indices | Phase 22 RVQ, epoch 197 |
| AVIF, up to 512 bytes | Complete independently decodable AVIF file | Conventional AV1 compression |

## Launch

```bash
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. The server binds to loopback by default (set `GRADIO_SERVER_NAME=0.0.0.0` to serve on a network, e.g. when hosting); Gradio public share links are disabled. Pillow 11.3+ with AVIF support is required. The app uses local system fonts and disables Gradio analytics.

The three learned codecs load `best.pt` (full encoder + decoder), `decoder.pt` (decoder-only export) and `training_completion.json` from `weights/face24_64/`, `weights/face24_128/`, and `weights/face24_256_rvq/`. Startup checks file hashes, selected-checkpoint metadata, parameter counts, and exact codebook/decoder tensor equality. Phase 22 checks both codebooks.

Face detection uses `weights/mediapipe/face_landmarker.task`, verified against a SHA-256 recorded in `demo/codec.py`. Encoders use CUDA when available, otherwise CPU; neural decoder exports run on CPU. Downloads are saved under `output/` (git-ignored).

## Use the comparison

1. In **Compare & encode**, choose one single-face JPG or PNG and click **Compare codecs**.
2. Compare the shared aligned input and four reconstructions. Every codec receives identical grayscale pixels. Previews use lossless PNG and nearest-neighbor display without enhancement.
   The original and shared input use matching size cards: the original shows the uploaded file’s exact byte count and dimensions; the shared input shows 5,120 uncompressed pixel bytes (64 × 80 × 1 byte), excluding file headers. These labels distinguish source-file size, raw pixels, and compressed QR payloads. The original can have different dimensions and color channels; preview PNG sizes are not used.
3. Click a preview or its **Show QR** button. This selects its cached original payload, updates the export selector and QR, and scrolls to the result. Changing the selector does the same. Nothing re-encodes a reconstructed preview.
4. Read the AVIF byte accounting below the cards; its card shows the actual complete-file size. Download the complete `.avif` independently if desired.
5. Download the QR PNG. In **Decode a QR**, upload it and click **Decode QR** to recover the face. The format is detected automatically.

Payloads are private to each Gradio browser session. Changing the photograph or an encoding failure clears previous results. Encode the photograph before choosing a preview. Binary face QR codes are not text/URLs; a phone camera may not offer a useful action. This demo's decoder reads raw barcode bytes.

## Honest AVIF accounting

AVIF receives the same 80×64 grayscale crop, with no resizing, color, augmentation or enhancement. The encoder uses monochrome 4:0:0, libavif speed 4, and one encoder thread. It searches quality 100 down to 0 and uses the highest numerical quality whose **entire file** fits 512 bytes. Descending search does not assume size is monotonic. If nothing fits, it reports failure rather than exceeding the limit.

The displayed identity is:

`total file bytes = complete AV1 bitstream bytes + file-container overhead bytes`

The bitstream count is the content of the ISO BMFF `mdat` box(es), including AV1 sequence/frame headers. Container overhead is every remaining byte, including box headers and metadata. Both are measured from the exact file; every byte is inside the QR. The budget is a maximum, not padding to 512 bytes. The downloadable `.avif` and QR-recovered file are byte-identical.

Learned options count raw indices and assume pre-shared neural decoder weights. AVIF counts its complete file and assumes a standard AVIF decoder. Decoder software and neural model weights are not transmitted or included for either option. QR mode/length bits, framing, and error correction add transport overhead for all options and are excluded from the displayed payload totals. Larger payloads can require denser QR codes.

All image payloads use QR byte mode directly, with medium error correction and a four-module quiet zone. There is no Base64, hexadecimal encoding, or model identifier. Every generated QR is scanned before being exposed. For qrcode's all-zero Reed–Solomon-block edge case, multiple standard byte-mode segments preserve the same recovered payload.

The decoder recognizes the AVIF file signature and validates a single 64×80 still image up to 512 bytes. Learned formats dispatch by exact 64/128/256-byte length. Other data is rejected. Length dispatch does not authenticate neural payloads or distinguish future codebooks with the same size. RVQ is stage-major, with each 16×8 grid serialized row-major. Its first 128 bytes are not Phase 21 compatible.

## Alignment and limitations

Uploaded photographs use the existing MediaPipe detector: paired eye corners estimate eye centers, with nose and mouth corners providing five points. The demo applies the frozen Face24 B similarity crop, 24-pixel interocular distance, eye midpoint (31.5, 23), gray padding and existing 4× cubic/Lanczos rendering. There is no matting or enhancement. Missing/multiple faces, implausible landmarks and critical crop clipping produce explanatory errors. Estimated upload landmarks differ from CelebA annotations.

This is an informal visual comparison, not a blinded human-recognition study. Model training and AVIF use different compression objectives. Quality is image dependent; AVIF's complete-file overhead can be substantial at this small resolution.

Page/config responses use `Cache-Control: no-store`; the manifest is served without enabling PWA/offline caching. Refresh open tabs after an update. Gradio 5.50's `readable_stream().close()` may log `Method not implemented.` at `stream.ts:186`; those upstream diagnostics remain visible.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# Optional: also run the end-to-end real-photo tests
FACE24_TEST_PHOTO=path/to/single_face.jpg python -m pytest -q
```

Tests cover neural byte order and export agreement, checkpoint tampering, binary QR patterns, retired-length rejection, real-photo workflows, AVIF complete-file decoding without neural models, exact size accounting, malformed/oversized AVIF rejection, bounded quality selection, and exporting cached previews without another encoding pass.
