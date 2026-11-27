"""Header-stripped AVIF baseline vs the Face24 codecs on the held-out test faces.

Only the AV1 frame data is counted as payload. Everything else in the file (the ISO BMFF
container, temporal delimiter and sequence header) is a fixed template the decoder keeps,
and the script checks that the template really is identical for every face and quality.

    python tools/avif_eval.py --test-manifest E:/eigenfaces/face24_dataset/manifests/accepted_test.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QUALITIES = range(0, 101)
BUDGETS = (64, 128, 256)


def encode_full(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "AVIF", quality=quality, speed=4, subsampling="4:0:0", max_threads=1)
    return buffer.getvalue()


def leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def split(avif: bytes):
    """-> (container bytes before mdat, list of (obu_type, obu_header, obu_body))."""
    start = avif.index(b"mdat") - 4
    data, p, obus = avif[start + 8:], 0, []
    while p < len(data):
        header_start, kind = p, (data[p] >> 3) & 0xF
        if data[p] & 0x04:
            raise ValueError("unexpected OBU extension header")
        p += 1
        size = shift = 0
        while True:
            byte = data[p]
            p += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if byte < 0x80:
                break
        obus.append((kind, data[header_start:p], data[p:p + size]))
        p += size
    return avif[:start], obus


def strip(avif: bytes):
    """-> (frame payload, template key). The key covers every byte the decoder must supply."""
    container, obus = split(avif)
    if [o[0] for o in obus] != [2, 1, 6]:
        raise ValueError(f"unexpected OBU layout {[o[0] for o in obus]}")
    fixed = bytearray(container)
    fixed[115:117] = b"\0\0"  # iloc extent length: derived from the payload size
    key = bytes(fixed) + obus[0][1] + obus[1][1] + obus[1][2] + obus[2][1][:1]
    return obus[2][2], key


def rebuild(template: bytes, payload: bytes) -> bytes:
    container, obus = split(template)
    mdat = obus[0][1] + obus[1][1] + obus[1][2] + obus[2][1][:1] + leb128(len(payload)) + payload
    head = bytearray(container)
    head[115:117] = len(mdat).to_bytes(2, "big")
    return bytes(head) + (len(mdat) + 8).to_bytes(4, "big") + b"mdat" + mdat


def one_face(path: str):
    image = Image.open(path).convert("L")
    target = np.asarray(image, np.float64)
    keys, best = set(), {}
    for quality in QUALITIES:
        payload, key = strip(encode_full(image, quality))
        keys.add(hashlib.sha256(key).hexdigest())
        for budget in BUDGETS:
            if len(payload) <= budget and (budget not in best or quality > best[budget][0]):
                best[budget] = (quality, payload)
    out = {}
    for budget, (quality, payload) in best.items():
        decoded = np.asarray(Image.open(io.BytesIO(rebuild(TEMPLATE, payload))).convert("L"), np.float64)
        out[budget] = {"quality": quality, "bytes": len(payload), "mse": float(((decoded - target) ** 2).mean() / 255 ** 2)}
    return out, sorted(keys)


TEMPLATE = encode_full(Image.new("L", (64, 80), 128), 50)


def codec_mse(pixels: np.ndarray):
    import torch
    from face24 import codec128, codec256, codec64
    sys.path.insert(0, str(ROOT / "tools"))
    from export_web import MODELS, WEIGHTS, encode_decode
    del codec64, codec128, codec256
    target = pixels.astype(np.float32) / 255
    result = {}
    with torch.no_grad():
        for size, (directory, load_full) in MODELS.items():
            model, _ = load_full(WEIGHTS / directory / "best.pt", "cpu")
            model.eval()
            recon = np.concatenate([encode_decode(model, size, pixels[i:i + 250])[1] for i in range(0, len(pixels), 250)])
            result[size] = ((np.clip(recon, 0, 1) - target) ** 2).reshape(len(pixels), -1).mean(1)
    return result


def psnr(mse):
    return 10 * np.log10(1 / np.asarray(mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "models" / "avif_eval.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.test_manifest.read_text(encoding="utf-8").splitlines()[:args.count]]
    paths = [r["output_path"] for r in rows]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one_face, paths, chunksize=8))
    keys = sorted({k for _, ks in results for k in ks})
    pixels = np.stack([np.asarray(Image.open(p).convert("L")) for p in paths])
    codec = codec_mse(pixels)
    report = {"images": len(paths), "distinct_templates": len(keys), "budgets": {}}
    for budget in BUDGETS:
        fits = [r[budget] for r, _ in results if budget in r]
        avif = np.array([r[budget]["mse"] if budget in r else np.nan for r, _ in results])
        ok = ~np.isnan(avif)
        report["budgets"][budget] = {
            "avif_fits_fraction": float(ok.mean()),
            "avif_mean_bytes": float(np.mean([f["bytes"] for f in fits])),
            "avif_mean_quality": float(np.mean([f["quality"] for f in fits])),
            "avif_psnr_of_mean_mse": float(psnr(avif[ok].mean())),
            "avif_mean_psnr": float(psnr(avif[ok]).mean()),
            "codec_psnr_of_mean_mse": float(psnr(codec[budget].mean())),
            "codec_mean_psnr": float(psnr(codec[budget]).mean()),
            "avif_better_fraction": float((avif[ok] < codec[budget][ok]).mean()),
            "avif_minus_codec_db_p10_p50_p90": [float(x) for x in np.percentile(psnr(avif[ok]) - psnr(codec[budget][ok]), [10, 50, 90])],
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
