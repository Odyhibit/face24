# Face24 — a face in a QR code

Tiny learned image codecs that squeeze an aligned 64×80 grayscale face into **64, 128, or 256 bytes**: small enough to fit in a single QR code. The web demo encodes a photo with all three neural codecs and a conventional AVIF baseline (≤ 512 bytes). It shows the reconstructions side by side and exports or decodes the QR codes.

This is an educational, non-commercial research project. The payloads are lossy compressed pictures, not identity verification, biometrics, or encryption.

## Quick start

```bash
git clone https://github.com/Odyhibit/eigenfaces.git
cd eigenfaces
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860, upload a photo containing one face, and click **Compare codecs**. Runs on CPU; a CUDA GPU is used for encoding when available. Python 3.10–3.13.

## The codecs

| Payload | Model | How the bytes are used | SSIM ↑ | LPIPS ↓ |
|---|---|---|---:|---:|
| 64 B | VQ-VAE, 8×8 grid | one codebook index (0–255) per cell | 0.838 | 0.056 |
| 128 B (default) | VQ-VAE, 16×8 grid | one codebook index per cell | 0.883 | 0.046 |
| 256 B | Residual VQ, 2 × 16×8 | coarse + residual index per cell | 0.920 | 0.033 |
| ≤ 512 B | AVIF (AV1) baseline | complete standalone `.avif` file | — | — |

Metrics are on the 7,557-image CelebA test split at 64×80 grayscale.

Each neural model is a convolutional encoder → vector quantizer (256-entry codebook) → convolutional decoder with ~4M parameters, trained with pixel, facial-region, LPIPS, and ArcFace identity losses. The QR contains only the raw codebook indices: no header, Base64, or model ID. The decoder weights are the "shared secret" both sides already have, and the decoder picks the model from the payload length.

Photos are aligned with MediaPipe Face Landmarker: a similarity transform puts the eyes 24 px apart at a fixed position, then the crop is rendered to 64×80 grayscale.

## Repository layout

```
app.py                 entry point: python app.py
face24/                model package (inference only)
  layers.py            shared VQ encoder / quantizer blocks
  codec64.py           64-byte codec
  codec128.py          128-byte codec
  codec256.py          256-byte residual-VQ codec
  alignment.py         landmark-based Face24 crop
demo/                  Gradio web app
  codec.py             model loading + integrity checks, QR encode/decode
  avif_codec.py        AVIF baseline with exact byte accounting
  interface.py         UI layout;  styles.css, frontend.py
weights/               trained checkpoints (~88 MB) + MediaPipe landmarker
tests/                 pytest suite
docs/DEMO.md           detailed demo behaviour and byte-accounting notes
docs/HOSTING.md        options for putting the demo online
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Set `FACE24_TEST_PHOTO=/path/to/single_face.jpg` to also run the end-to-end real-photo tests.

## Data, weights, and licences

- Code: MIT (see `LICENSE`).
- The codec weights were trained on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is licensed for **non-commercial research only**. Use the weights accordingly. The dataset itself is not included.
- `weights/mediapipe/face_landmarker.task` is Google's [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) model (Apache-2.0).
- Training code and experiment history live in a separate research workspace and are not part of this repository.
