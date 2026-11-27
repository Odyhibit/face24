#!/usr/bin/env python3
"""
Convert the face24 web models (web/models/face24_{64,128,256}.bin) into the
device format used by face24-core2, and write the self-test vectors.

    python3 tools/make_device_models.py [path/to/face24]   (default: the repo this file is in)

Writes:
    data/models.bin       -> flash to the "models" partition (see README.md)
    src/testvectors.h     -> expected decoder outputs from PyTorch (int8-simulated)

Only the decoder and codebooks are kept; the encoder is not needed on the device.
Weights stay int8 with one float scale per output channel, exactly as in the
web export, so the device reproduces the quality measured in int8_eval.json.

Device model layout (little-endian):
  header   12 x u32: magic 'F24D', version, payload_bytes, kind (0 vq, 1 rvq),
                     grid_h, grid_w, n_tensors, n_ops, codebook_a, codebook_b,
                     data_offset, reserved
  tensors  n x 9 u32: dtype (0 f32, 1 i8), ndim, shape[4], offset, scale_offset, scale_axis
  ops      n x 8 i32: type, w, b, stride_h, stride_w, pad_h, pad_w, size (h<<16 | w)
  data     16-byte aligned; offsets are relative to data_offset
Pack layout: 'F24P', count, then count x (payload_bytes, offset, size); each
model starts on a 64 KB boundary so it can be memory-mapped on its own.
"""
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

OP = {"conv": 1, "convT": 2, "relu": 3, "sigmoid": 4, "up2": 5, "upsize": 6, "res_begin": 7, "res_end": 8}
NONE = 0xFFFFFFFF
ALIGN_MODEL = 0x10000


def read_web_model(path):
    raw = path.read_bytes()
    if raw[:4] != b"F24W":
        raise ValueError(f"{path} is not a face24 web model")
    hlen = struct.unpack_from("<I", raw, 4)[0]
    header = json.loads(raw[8:8 + hlen])
    return header, raw[8 + hlen:]


def flatten(ops, out):
    for op in ops:
        kind = op["op"]
        if kind == "conv":
            out.append((OP["conv"], op["w"], op["b"], *op["stride"], *op["pad"], 0))
        elif kind == "convT":
            out.append((OP["convT"], op["w"], op["b"], 1, 1, 0, 0, 0))
        elif kind in ("relu", "sigmoid"):
            out.append((OP[kind], None, None, 0, 0, 0, 0, 0))
        elif kind == "upsample":
            if op.get("size"):
                h, w = op["size"]
                out.append((OP["upsize"], None, None, 0, 0, 0, 0, (h << 16) | w))
            else:
                assert op.get("scale", 2) == 2
                out.append((OP["up2"], None, None, 0, 0, 0, 0, 0))
        elif kind == "residual":
            out.append((OP["res_begin"], None, None, 0, 0, 0, 0, 0))
            flatten(op["body"], out)
            out.append((OP["res_end"], None, None, 0, 0, 0, 0, 0))
        else:
            raise ValueError(f"unsupported op {kind}")


def build_device_model(header, blob):
    ops = []
    flatten(header["decoder"], ops)
    used = [n for n in header["tensors"] if n.startswith("decoder.")]
    codebooks = ["codebook1", "codebook2"] if header["kind"] == "rvq" else ["codebook"]
    names = used + codebooks
    index = {n: i for i, n in enumerate(names)}

    data = bytearray()

    def add(buf):
        while len(data) % 16:
            data.append(0)
        off = len(data)
        data.extend(buf)
        return off

    table = []
    for name in names:
        t = header["tensors"][name]
        shape = list(t["shape"]) + [1] * (4 - len(t["shape"]))
        count = int(np.prod(t["shape"]))
        if t["dtype"] == "float32":
            arr = np.frombuffer(blob, np.float32, count, t["offset"])
            table.append((0, len(t["shape"]), *shape, add(arr.tobytes()), NONE, NONE))
        else:
            q = np.frombuffer(blob, np.int8, count, t["offset"])
            nscale = t["shape"][t["scale_axis"]]
            sc = np.frombuffer(blob, np.float32, nscale, t["scale_offset"])
            table.append((1, len(t["shape"]), *shape, add(q.tobytes()), add(sc.tobytes()), t["scale_axis"]))

    op_rows = []
    for row in ops:
        typ, w, b, *rest = row
        op_rows.append((typ, index[w] if w else -1, index[b] if b else -1, *rest))

    n_t, n_o = len(table), len(op_rows)
    data_offset = 48 + n_t * 36 + n_o * 32
    data_offset = (data_offset + 15) & ~15
    out = bytearray(struct.pack(
        "<4sIIIIIIIIIII", b"F24D", 1, header["payload_bytes"], 1 if header["kind"] == "rvq" else 0,
        header["grid"][0], header["grid"][1], n_t, n_o,
        index[codebooks[0]], index[codebooks[1]] if len(codebooks) > 1 else NONE, data_offset, 0))
    for row in table:
        out += struct.pack("<9I", *row)
    for row in op_rows:
        out += struct.pack("<8i", *row)
    out += b"\0" * (data_offset - len(out))
    out += data
    return bytes(out)


def c_array(name, ctype, values, per_line=12):
    lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(values[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    here = Path(__file__).resolve().parent.parent  # firmware/face24-core2
    repo = Path(sys.argv[1]) if len(sys.argv) == 2 else here.parent.parent
    (here / "data").mkdir(exist_ok=True)

    models = []
    for size in (64, 128, 256):
        header, blob = read_web_model(repo / "web" / "models" / f"face24_{size}.bin")
        models.append((size, build_device_model(header, blob)))

    pack = bytearray(struct.pack("<4sI", b"F24P", len(models)))
    dir_off = len(pack)
    pack += b"\0" * (12 * len(models))
    entries = []
    for size, blob in models:
        pad = (-len(pack)) % ALIGN_MODEL
        pack += b"\0" * pad
        entries.append((size, len(pack), len(blob)))
        pack += blob
    for i, e in enumerate(entries):
        struct.pack_into("<3I", pack, dir_off + 12 * i, *e)
    (here / "data" / "models.bin").write_bytes(pack)
    for size, off, n in entries:
        print(f"face24_{size}: {n / 1e6:.2f} MB at offset 0x{off:06X}")
    print(f"data/models.bin: {len(pack) / 1e6:.2f} MB")

    vectors = json.loads((repo / "web" / "testvectors.json").read_text())
    parts = ["// Generated by tools/make_device_models.py from face24 web/testvectors.json.",
             "// Expected decoder outputs come from the int8-simulated PyTorch models.",
             "#pragma once", "#include <stdint.h>", ""]
    rows = []
    for size in (64, 128, 256):
        v = vectors["models"][str(size)]
        for k, (p, r) in enumerate(zip(v["payloads"], v["recon"])):
            payload = np.frombuffer(base64.b64decode(p), np.uint8)
            recon = np.frombuffer(base64.b64decode(r), np.float32)
            assert payload.size == size and recon.size == 80 * 64
            parts.append(c_array(f"tv{size}_{k}_payload", "uint8_t", [str(x) for x in payload]))
            parts.append(c_array(f"tv{size}_{k}_recon", "float", [f"{x:.8g}f" for x in recon], 8))
            rows.append(f"    {{{size}, tv{size}_{k}_payload, tv{size}_{k}_recon}},")
    parts += ["", "struct TestVector { int size; const uint8_t* payload; const float* recon; };",
              "static const TestVector TEST_VECTORS[] = {", *rows, "};", ""]
    (here / "src" / "testvectors.h").write_text("\n".join(parts))
    print(f"src/testvectors.h: {len(rows)} vectors")


if __name__ == "__main__":
    main()
