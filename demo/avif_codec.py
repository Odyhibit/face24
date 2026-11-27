"""Header-stripped AVIF baseline, matching the browser demo (web/avif.js, web/worker.js).

Only the AV1 frame data counts as AVIF's payload. Everything else in the file (ISO BMFF
container, temporal delimiter, sequence header; about 280 bytes) is identical for every face,
so the decoder keeps it as a fixed template, the same way the Face24 decoder keeps its codebook.
For each codec size, AVIF gets the smallest frame that is at least as many bytes.
"""
from __future__ import annotations

import base64
import io

from PIL import Image, features

BUDGETS = (64, 128, 256)
QUALITIES = range(0, 101)

# Identical to TEMPLATE_B64 in web/avif.js (tests check they stay in sync). Made with Pillow:
# a flat grey 64x80 image, quality 50, speed 4, 4:0:0.
TEMPLATE = base64.b64decode(
    "AAAAHGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZgAAAOltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABwaWN0AAAAAAAAAAAAAAAAAAAAAA5waXRt"
    "AAAAAAABAAAAHmlsb2MAAAAARAAAAQABAAAAAQAAAQ0AAAATAAAAKGlpbmYAAAAAAAEAAAAaaW5mZQIAAAAAAQAAYXYwMUNvbG9yAAAAAGhp"
    "cHJwAAAASWlwY28AAAAUaXNwZQAAAAAAAABAAAAAUAAAAA5waXhpAAAAAAEIAAAADGF2MUOBABwAAAAAE2NvbHJuY2x4AAEADQAGgAAAABdp"
    "cG1hAAAAAAAAAAEAAQQBAoMEAAAAG21kYXQSAAoGGBW/nsqgMgcUAAAAiBDA")
ILOC_LENGTH = 115  # 2 bytes, big-endian: length of the mdat contents


def _leb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _split(avif: bytes):
    """-> (container bytes before mdat, [(obu_type, obu_header, obu_body), ...])."""
    position = 0
    while position + 8 <= len(avif):
        size = int.from_bytes(avif[position:position + 4], "big")
        if avif[position + 4:position + 8] == b"mdat":
            break
        if size < 8:
            raise ValueError("Invalid AVIF box length.")
        position += size
    else:
        raise ValueError("AVIF file has no mdat box.")
    data, p, obus = avif[position + 8:], 0, []
    while p < len(data):
        start, kind = p, (data[p] >> 3) & 0xF
        if data[p] & 0x04:
            raise ValueError("Unexpected AV1 OBU extension header.")
        if not data[p] & 0x02:
            raise ValueError("AV1 OBU without a size field.")
        p += 1
        size = shift = 0
        while True:
            if p >= len(data):
                raise ValueError("Truncated AV1 OBU size.")
            byte = data[p]
            p += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if byte < 0x80:
                break
        obus.append((kind, data[start:p], data[p:p + size]))
        p += size
    return avif[:position], obus


_CONTAINER, _OBUS = _split(TEMPLATE)
_FIXED_OBUS = _OBUS[0][1] + _OBUS[1][1] + _OBUS[1][2]  # temporal delimiter + sequence header
_FRAME_HEADER = _OBUS[2][1][:1]
# Sequence headers whose frames decode identically under the template's header (see web/avif.js:
# the browser encoder also writes colour labels that don't affect a full-range 4:0:0 image).
_EQUIVALENT_HEADERS = {_OBUS[1][2].hex(), "1815bf9ecb020206a0"}


def strip(avif: bytes) -> bytes:
    """Complete AVIF file -> frame payload. Raises if the file doesn't fit the shared template."""
    _, obus = _split(avif)
    if [o[0] for o in obus] != [2, 1, 6]:
        raise ValueError(f"Unexpected AV1 layout (OBU types {[o[0] for o in obus]}).")
    if obus[1][2].hex() not in _EQUIVALENT_HEADERS:
        raise ValueError("AV1 sequence header differs from the shared template.")
    if obus[2][1][:1] != _FRAME_HEADER:
        raise ValueError("Unexpected AV1 frame header.")
    return bytes(obus[2][2])


def rebuild(payload: bytes) -> bytes:
    """Frame payload -> complete AVIF file (template container + this frame)."""
    mdat = _FIXED_OBUS + _FRAME_HEADER + _leb128(len(payload)) + bytes(payload)
    head = bytearray(_CONTAINER)
    head[ILOC_LENGTH:ILOC_LENGTH + 2] = len(mdat).to_bytes(2, "big")
    return bytes(head) + (len(mdat) + 8).to_bytes(4, "big") + b"mdat" + mdat


def encode_frame(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "AVIF", quality=quality, speed=4, subsampling="4:0:0", max_threads=1)
    return strip(buffer.getvalue())


def decode(payload: bytes) -> Image.Image:
    """Frame payload -> 64x80 grayscale image."""
    try:
        with Image.open(io.BytesIO(rebuild(payload))) as image:
            if image.format != "AVIF" or image.size != (64, 80):
                raise ValueError("Expected one 64×80 AVIF face image.")
            image.load()
            return image.convert("L")
    except (OSError, SyntaxError, RuntimeError) as exc:  # Pillow raises RuntimeError for corrupt AV1 data
        raise ValueError("The AVIF frame could not be decoded.") from exc


def encode_for_budgets(image: Image.Image, budgets=BUDGETS) -> dict:
    """For each budget, the smallest frame that is at least that many bytes (ties: higher quality),
    so AVIF never gets fewer bytes than the codec. Sizes aren't strictly monotonic in quality, so
    scan upwards and stop once frames are well past the largest budget. Budgets that no quality
    reaches are left out.  -> {budget: {"quality": q, "payload": bytes}}"""
    if not features.check("avif"):
        raise RuntimeError("This demo needs Pillow with AVIF support (Pillow 11.3 or later).")
    if image.size != (64, 80):
        raise ValueError("Expected a 64×80 aligned grayscale face.")
    image = image.convert("L")
    found, stop = {}, 2 * max(budgets)
    for quality in QUALITIES:
        payload = encode_frame(image, quality)
        for budget in budgets:
            if len(payload) >= budget and (budget not in found or len(payload) <= len(found[budget]["payload"])):
                found[budget] = {"quality": quality, "payload": payload}
        if len(payload) > stop:
            break
    return found


def caption(found: dict, budget: int) -> str:
    entry = found.get(budget)
    if entry is None:
        return "**AVIF** · no frame this large"
    return f"**AVIF · {len(entry['payload'])} bytes** · quality {entry['quality']}"


def description(found: dict) -> str:
    parts = [f"{budget} B → AVIF {len(found[budget]['payload'])} B (quality {found[budget]['quality']})"
             if budget in found else f"{budget} B → no AVIF frame this large" for budget in BUDGETS]
    return "**This photo:** " + " · ".join(parts) + "."
