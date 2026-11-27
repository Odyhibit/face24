"""Export the Face24 codecs as int8-weight files for the browser demo (web/).

Weights: symmetric int8, one scale per output channel (conv / conv-transpose).
Biases and codebooks stay float32 (they are tiny and argmin is sensitive to them).
Activations are float32 in the browser.

    python tools/export_web.py                       # export models + test vectors
    python tools/export_web.py --eval 1000 \\
        --test-manifest E:/eigenfaces/face24_dataset/manifests/accepted_test.jsonl

File format (little endian): b"F24W", uint32 header length, UTF-8 JSON header,
zero padding to a 4-byte boundary, then tensor data (each tensor 4-byte aligned).
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face24 import codec128, codec256, codec64  # noqa: E402
from face24.layers import Residual  # noqa: E402

WEIGHTS = ROOT / "weights"
OUT = ROOT / "web" / "models"
MODELS = {
    64: ("face24_64", codec64.load_full),
    128: ("face24_128", codec128.load_full),
    256: ("face24_256_rvq", codec256.load_full),
}


# ---------------------------------------------------------------- graph export
def describe(seq: nn.Sequential, prefix: str, ops: list, params: dict):
    for i, m in enumerate(seq):
        name = f"{prefix}.{i}"
        if isinstance(m, nn.Conv2d):
            assert m.dilation == (1, 1) and m.groups == 1
            ops.append({"op": "conv", "w": name + ".weight", "b": name + ".bias",
                        "stride": list(m.stride), "pad": list(m.padding)})
            params[name + ".weight"] = (m.weight, 0)
            params[name + ".bias"] = (m.bias, None)
        elif isinstance(m, nn.ConvTranspose2d):
            assert m.stride == (1, 1) and m.padding == (0, 0) and m.groups == 1
            ops.append({"op": "convT", "w": name + ".weight", "b": name + ".bias"})
            params[name + ".weight"] = (m.weight, 1)  # [in, out, kh, kw]: quantize per out channel
            params[name + ".bias"] = (m.bias, None)
        elif isinstance(m, nn.ReLU):
            ops.append({"op": "relu"})
        elif isinstance(m, nn.Sigmoid):
            ops.append({"op": "sigmoid"})
        elif isinstance(m, nn.Upsample):
            assert m.mode == "bilinear" and not m.align_corners
            if m.size is not None:
                ops.append({"op": "upsample", "size": list(m.size)})
            else:
                assert float(m.scale_factor) == 2.0
                ops.append({"op": "upsample", "scale": 2})
        elif isinstance(m, Residual):
            body = []
            describe(m.layers, name + ".layers", body, params)
            ops.append({"op": "residual", "body": body})
        else:
            raise TypeError(f"Unsupported layer {type(m).__name__} at {name}")


def quantize(w: torch.Tensor, axis: int):
    w = w.detach().float().cpu().numpy()
    moved = np.moveaxis(w, axis, 0)
    scale = np.abs(moved.reshape(moved.shape[0], -1)).max(1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(moved / scale.reshape(-1, *[1] * (moved.ndim - 1))), -127, 127).astype(np.int8)
    deq = np.moveaxis(q.astype(np.float32) * scale.reshape(-1, *[1] * (moved.ndim - 1)), 0, axis)
    return np.moveaxis(q, 0, axis), scale.astype(np.float32), deq


def export(size: int):
    directory, load_full = MODELS[size]
    model, _ = load_full(WEIGHTS / directory / "best.pt", "cpu")
    enc_ops, dec_ops, params = [], [], {}
    describe(model.encoder.layers, "encoder.layers", enc_ops, params)
    describe(model.decoder.layers, "decoder.layers", dec_ops, params)
    if size == 256:
        params["codebook1"] = (model.quantizer.embedding1.weight, None)
        params["codebook2"] = (model.quantizer.embedding2.weight, None)
    else:
        params["codebook"] = (model.quantizer.embedding.weight, None)

    tensors, blobs, offset = {}, [], 0
    simulated = copy.deepcopy(model)  # same model with int8-dequantized weights, for evaluation
    sim_params = dict(simulated.named_parameters())

    def add(data: np.ndarray):
        nonlocal offset
        raw = data.tobytes(order="C")
        raw += b"\0" * (-len(raw) % 4)
        blobs.append(raw)
        start = offset
        offset += len(raw)
        return start

    for name, (tensor, axis) in params.items():
        torch_name = {"codebook": "quantizer.embedding.weight", "codebook1": "quantizer.embedding1.weight",
                      "codebook2": "quantizer.embedding2.weight"}.get(name, name)
        if axis is None:
            data = tensor.detach().float().cpu().numpy().astype(np.float32)
            tensors[name] = {"dtype": "float32", "shape": list(data.shape), "offset": add(data)}
        else:
            q, scale, deq = quantize(tensor, axis)
            tensors[name] = {"dtype": "int8", "shape": list(q.shape), "offset": add(q),
                             "scale_axis": axis, "scale_offset": add(scale)}
            with torch.no_grad():
                sim_params[torch_name].copy_(torch.from_numpy(deq))

    grid = [8, 8] if size == 64 else [16, 8]
    header = {"format": 1, "payload_bytes": size, "kind": "rvq" if size == 256 else "vq",
              "grid": grid, "image": [80, 64], "encoder": enc_ops, "decoder": dec_ops, "tensors": tensors}
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-(8 + len(head)) % 4)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"face24_{size}.bin"
    path.write_bytes(b"F24W" + struct.pack("<I", len(head)) + head + b"".join(blobs))
    return model.eval(), simulated.eval(), path


# ---------------------------------------------------------------- inference helpers
@torch.inference_mode()
def encode_decode(model, size, pixels: np.ndarray):
    """pixels: N x 80 x 64 uint8 -> (indices N x ..., recon N x 80 x 64 float)."""
    x = torch.from_numpy(pixels.astype(np.float32)[:, None] / 255)
    z = model.encoder(x)
    indices = model.quantizer.indices(z)
    if size == 256:
        q = model.quantizer.embedding1(indices[:, 0]) + model.quantizer.embedding2(indices[:, 1])
    else:
        q = model.quantizer.embedding(indices)
    recon = model.decoder(q.permute(0, 3, 1, 2).contiguous())[:, 0]
    return indices.numpy().astype(np.uint8), recon.numpy()


def evaluate(models, manifest: Path, count: int):
    from PIL import Image
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()[:count]]
    pixels = np.stack([np.asarray(Image.open(r["output_path"]).convert("L")) for r in rows])
    target = pixels.astype(np.float32) / 255
    report = {"images": len(rows), "models": {}}
    for size, (full, sim, _) in models.items():
        stats = {}
        idx_f, rec_f = encode_decode(full, size, pixels)
        idx_q, rec_q = encode_decode(sim, size, pixels)
        stats["mse_float"] = float(((rec_f - target) ** 2).mean())
        stats["mse_int8"] = float(((rec_q - target) ** 2).mean())
        stats["psnr_float"] = float(10 * np.log10(1 / stats["mse_float"]))
        stats["psnr_int8"] = float(10 * np.log10(1 / stats["mse_int8"]))
        stats["index_agreement"] = float((idx_f == idx_q).mean())
        stats["payload_identical_fraction"] = float((idx_f == idx_q).reshape(len(rows), -1).all(1).mean())
        stats["mean_abs_recon_change_255"] = float(np.abs(rec_f - rec_q).mean() * 255)
        report["models"][size] = stats
    return report


def test_vectors(models, photo: Path | None):
    """Reference outputs from the int8-simulated PyTorch models, for checking the JS engine."""
    rng = np.random.default_rng(7)
    inputs = [rng.integers(0, 256, (80, 64), dtype=np.uint8)]
    ys, xs = np.mgrid[0:80, 0:64]
    inputs.append((127 + 100 * np.sin(xs / 6.0) * np.cos(ys / 9.0)).astype(np.uint8))
    vectors = {"inputs": [base64.b64encode(p.tobytes()).decode() for p in inputs], "models": {}}
    for size, (_, sim, _) in models.items():
        idx, rec = encode_decode(sim, size, np.stack(inputs))
        vectors["models"][size] = {
            "payloads": [base64.b64encode(i.tobytes(order="C")).decode() for i in idx],
            "recon": [base64.b64encode(r.astype(np.float32).tobytes()).decode() for r in rec],
        }
    if photo is not None:
        # Kept out of git: it embeds the photo, which usually can't be redistributed.
        local = {"alignment": alignment_vector(photo)}
        (ROOT / "web" / "testvectors_local.json").write_text(json.dumps(local), encoding="utf-8")
    return vectors


def alignment_vector(photo: Path):
    """Python landmarks + aligned crop for one photo, so the JS aligner can be compared exactly."""
    from PIL import Image, ImageOps

    from demo import codec
    image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    aligner = codec.Face24Aligner()
    rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    result = aligner._detector.detect(aligner._mp.Image(image_format=aligner._mp.ImageFormat.SRGB, data=rgb))
    aligner.close()
    mesh = np.asarray([[p.x, p.y] for p in result.face_landmarks[0]], dtype=np.float64)
    h, w = rgb.shape[:2]
    five = codec.five_points_from_mesh(result.face_landmarks[0], w, h)
    crop = codec.align_from_landmarks(image, five, mesh * [w, h])
    return {"width": w, "height": h, "rgb": base64.b64encode(rgb.tobytes()).decode(),
            "landmarks": mesh.round(7).tolist(), "aligned": base64.b64encode(np.asarray(crop).tobytes()).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=int, default=0, help="number of test crops to evaluate")
    parser.add_argument("--test-manifest", type=Path, help="Face24 accepted_test.jsonl (needed for --eval)")
    parser.add_argument("--photo", type=Path,
                        help="single-face photo for the alignment test vector (written to web/testvectors_local.json, git-ignored)")
    args = parser.parse_args()
    torch.set_num_threads(8)
    models = {size: export(size) for size in MODELS}
    for size, (_, _, path) in models.items():
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.2f} MB")
    (ROOT / "web" / "testvectors.json").write_text(json.dumps(test_vectors(models, args.photo)), encoding="utf-8")
    if args.eval:
        report = evaluate(models, args.test_manifest, args.eval)
        (OUT / "int8_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
