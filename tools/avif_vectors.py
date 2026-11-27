"""Reference vectors for the browser's header-stripped AVIF code (web/avif.js).

For each test input in web/testvectors.json and each budget, stores Pillow's frame payload,
the complete file Pillow wrote, and Pillow's decoded pixels.   Run: python tools/avif_vectors.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avif_eval import BUDGETS, QUALITIES, ROOT, encode_full, rebuild, strip, TEMPLATE  # noqa: E402

b64 = lambda b: base64.b64encode(bytes(b)).decode()


def face_like() -> np.ndarray:
    """A crude synthetic face (shaded oval, eyes, nose, mouth, slight noise) so the vectors
    include frames of realistic size without embedding a real person."""
    ys, xs = np.mgrid[0:80, 0:64].astype(np.float64)
    img = 60 + 40 * xs / 64
    oval = ((xs - 32) / 22) ** 2 + ((ys - 42) / 30) ** 2
    img = np.where(oval < 1, 175 - 25 * oval - 0.4 * (ys - 42), img)
    for cx in (20, 44):
        img -= 90 * np.exp(-(((xs - cx) / 4.5) ** 2 + ((ys - 30) / 2.5) ** 2))
        img -= 35 * np.exp(-(((xs - cx) / 7) ** 2 + ((ys - 24) / 1.5) ** 2))
    img -= 30 * np.exp(-(((xs - 32) / 3) ** 2 + ((ys - 45) / 7) ** 2))
    img -= 70 * np.exp(-(((xs - 32) / 9) ** 2 + ((ys - 58) / 2) ** 2))
    img += np.random.default_rng(3).normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("faces", nargs="*", type=Path, help="optional 64x80 face images to add")
    args = parser.parse_args()
    inputs = json.loads((ROOT / "web" / "testvectors.json").read_text(encoding="utf-8"))["inputs"]
    inputs.append(b64(face_like().tobytes()))
    for path in args.faces:  # optional real 64x80 faces; don't commit vectors made from images you can't share
        inputs.append(b64(Image.open(path).convert("L").tobytes()))
    out = {"template": b64(TEMPLATE), "inputs": inputs, "cases": []}
    for n, encoded in enumerate(inputs):
        image = Image.frombytes("L", (64, 80), base64.b64decode(encoded))
        for budget in BUDGETS:
            for quality in reversed(QUALITIES):
                full = encode_full(image, quality)
                payload, _ = strip(full)
                if len(payload) <= budget:
                    break
            else:
                continue
            rebuilt = rebuild(TEMPLATE, payload)
            decoded = np.asarray(Image.open(io.BytesIO(rebuilt)).convert("L"))
            out["cases"].append({"input": n, "budget": budget, "quality": quality, "payload": b64(payload),
                                 "rebuilt": b64(rebuilt), "decoded": b64(decoded.tobytes())})
    path = ROOT / "web" / "avif_vectors.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {len(out['cases'])} cases to {path}")


if __name__ == "__main__":
    main()
