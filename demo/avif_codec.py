"""Self-contained AVIF files: count every byte carried by the QR."""
from __future__ import annotations

import io

from PIL import Image, features

BUDGET = 512


def is_avif(payload: bytes) -> bool:
    return (len(payload) >= 16 and payload[4:8] == b"ftyp"
            and payload[8:12] == b"avif")


def accounting(payload: bytes) -> dict:
    """Separate the complete AV1 bitstream from ISO BMFF container bytes."""
    if not is_avif(payload) or len(payload) > BUDGET:
        raise ValueError("Expected a complete AVIF file of at most 512 bytes.")
    position, bitstream = 0, 0
    while position < len(payload):
        if len(payload) - position < 8:
            raise ValueError("Truncated AVIF container.")
        size = int.from_bytes(payload[position:position + 4], "big")
        kind = payload[position + 4:position + 8]
        header = 8
        if size == 1:
            if len(payload) - position < 16:
                raise ValueError("Truncated AVIF box length.")
            size = int.from_bytes(payload[position + 8:position + 16], "big")
            header = 16
        elif size == 0:
            size = len(payload) - position
        if size < header or position + size > len(payload):
            raise ValueError("Invalid AVIF box length.")
        if kind == b"mdat":
            bitstream += size - header
        position += size
    if not bitstream:
        raise ValueError("AVIF file contains no image bitstream.")
    return {"total_bytes": len(payload), "av1_bitstream_bytes": bitstream,
            "container_overhead_bytes": len(payload) - bitstream}


def decode(payload: bytes) -> Image.Image:
    accounting(payload)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "AVIF" or image.size != (64, 80) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected one 64×80 AVIF face image.")
            image.load()
            return image.convert("L")
    except (OSError, SyntaxError) as exc:
        raise ValueError("The AVIF file could not be decoded.") from exc


def encode(image: Image.Image) -> bytes:
    if not features.check("avif"):
        raise RuntimeError("This demo needs Pillow with AVIF support (Pillow 11.3 or later).")
    if image.size != (64, 80):
        raise ValueError("Expected a 64×80 aligned grayscale face.")
    image = image.convert("L")
    # Exhaustive descending quality avoids assuming that file sizes are monotonic.
    # At this tiny image size the bounded search is practical. Never exceed budget.
    for quality in range(100, -1, -1):
        buffer = io.BytesIO()
        image.save(buffer, "AVIF", quality=quality, speed=4, subsampling="4:0:0", max_threads=1)
        payload = buffer.getvalue()
        if len(payload) <= BUDGET:
            decode(payload)
            return payload
    raise ValueError("This face cannot fit in a complete 512-byte AVIF file, even at minimum quality.")


def description(payload: bytes) -> str:
    sizes = accounting(payload)
    return (f"**AVIF: {sizes['total_bytes']} bytes total** / 512-byte limit · "
            f"{sizes['av1_bitstream_bytes']} bytes AV1 image bitstream + "
            f"**{sizes['container_overhead_bytes']} bytes file-container overhead**. "
            "The bitstream includes its own decoding headers. Every file byte is included in the QR.")
