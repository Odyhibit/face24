# Face24 — a face in a QR code
## Try it now https://odyhibit.github.io/face24/web/
Tiny learned image codecs that squeeze an aligned 64×80 grayscale face into **64, 128, or 256 bytes**: small enough to fit in a single QR code. The web demo encodes a photo with all three neural codecs and a conventional AVIF baseline (≤ 512 bytes). It shows the reconstructions side by side and exports or decodes the QR codes.

This is an educational, non-commercial research project. The payloads are lossy compressed pictures, not identity verification, biometrics, or encryption.

## Quick start

```bash
git clone https://github.com/Odyhibit/face24.git
cd face24
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860, upload a photo containing one face, and click **Compare codecs**. Runs on CPU; a CUDA GPU is used for encoding when available. Python 3.10–3.13.

## In the browser

`web/` is a static version that runs entirely in the browser. 


```bash
python -m http.server 8765        # from the repository root
```

Then open http://localhost:8765/web/. It loads int8 copies of the three codecs (12 MB in total, versus 47 MB in float). Their quality is within 0.03 dB PSNR of the float models on 1,000 test faces. The page has three sections:

- **Encode:** MediaPipe finds the face, it's aligned exactly like the Python app, encoded by all three codecs, and turned into raw-byte QR codes. Each result is shown next to an AVIF image of at least the same byte count.
- **Decode:** read a QR code from an image or the camera. QR codes from the Python app work too.
- **Self-test:** checks the browser's results against PyTorch and Pillow reference outputs.

The AVIF comparison counts only the AV1 frame data. The rest of the file (about 280 bytes) is identical for every face, so `web/avif.js` keeps it as a template, the same way the codec decoder keeps its codebook. The AVIF encoder (libavif compiled to WebAssembly, about 3.5 MB) is loaded from jsdelivr only when needed.

Tools: `tools/export_web.py` writes the int8 models and test vectors; `node tools/check_web.mjs` compares the JavaScript with them; `tools/avif_eval.py` runs the AVIF-versus-codec comparison on the test split.

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
web/                   static in-browser version (int8 codecs, MediaPipe, QR, AVIF comparison)
tools/                 int8 export, JS-vs-PyTorch checks, AVIF evaluation
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

