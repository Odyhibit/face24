#!/usr/bin/env python3
"""
Generate the test QR codes for the BinaryScanTest sketch.

    pip install segno
    python3 make_test_qr.py [output_dir]

Every code is written in QR *byte mode* with no ECI header, so the symbol holds
exactly the raw bytes listed below. The sketch recognises each payload and
reports PASS/FAIL on screen.

  seq_256   bytes 0x00, 0x01, ... 0xFF  (every byte value once; the key test)
  rand_64   64 pseudo-random bytes   (fixed seed, matches the sketch's CRC table)
  rand_128  128 pseudo-random bytes
  rand_256  256 pseudo-random bytes
"""
import random
import sys
import zlib
from pathlib import Path

import segno


def payloads():
    yield "seq_256", bytes(range(256))
    for n in (64, 128, 256):
        rng = random.Random(1000 + n)
        yield f"rand_{n}", bytes(rng.randrange(256) for _ in range(n))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "qr_out")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in payloads():
        qr = segno.make(data, mode="byte", error="m", eci=False, boost_error=False, micro=False)
        path = out / f"{name}.png"
        qr.save(path, scale=8, border=4)
        print(f"{name:9s} len={len(data):3d} crc32=0x{zlib.crc32(data):08X} "
              f"version={qr.version} -> {path}")


if __name__ == "__main__":
    main()
