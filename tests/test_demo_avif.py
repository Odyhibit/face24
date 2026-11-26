import io

import numpy as np
import pytest
from PIL import Image

from demo import avif_codec
from demo.codec import DemoRuntime, decode_upload, encode_upload, make_binary_qr, preview_qr, read_binary_qr


@pytest.fixture(scope="module")
def face():
    # Structured gradients with edges exercise real image data at low bitrates.
    y, x = np.indices((80, 64))
    return Image.fromarray(((x * 3 + y * 2 + (x > 32) * 50) % 256).astype(np.uint8))


def test_avif_is_standalone_under_budget_and_accounting_adds_up(face):
    payload = avif_codec.encode(face)
    sizes = avif_codec.accounting(payload)
    assert sizes["total_bytes"] <= 512
    assert sizes["av1_bitstream_bytes"] + sizes["container_overhead_bytes"] == len(payload)
    assert sizes["container_overhead_bytes"] > 0 and sizes["av1_bitstream_bytes"] > 0
    with Image.open(io.BytesIO(payload)) as image:
        assert image.format == "AVIF" and image.size == (64, 80)
        image.load()
    assert str(sizes["total_bytes"]) in avif_codec.description(payload)
    assert str(sizes["container_overhead_bytes"]) in avif_codec.description(payload)
    # A decoder with no neural models still recovers this complete file.
    runtime = DemoRuntime.__new__(DemoRuntime)
    runtime.models = {}
    qr = make_binary_qr(payload)
    assert read_binary_qr(qr) == payload
    np.testing.assert_array_equal(np.asarray(runtime.reconstruct(payload)), np.asarray(avif_codec.decode(payload)))


@pytest.mark.parametrize("payload", [b"not avif", b"\0\0\0\x18ftypavif" + bytes(20), b"\0\0\0\x18ftypavif" + bytes(501)])
def test_corrupt_or_oversized_avif_is_rejected(payload):
    with pytest.raises(ValueError):
        avif_codec.decode(payload)


def test_highest_fitting_quality_without_assuming_monotonic_size(monkeypatch, face):
    tested = []
    def save(self, file, format, quality, **kwargs):
        tested.append(quality)
        file.write(bytes(512 if quality == 97 else 513))
    monkeypatch.setattr(Image.Image, "save", save)
    monkeypatch.setattr(avif_codec, "decode", lambda payload: face)
    assert len(avif_codec.encode(face)) == 512
    assert tested == [100, 99, 98, 97]


def test_no_fitting_file_fails_instead_of_exceeding_budget(monkeypatch, face):
    monkeypatch.setattr(Image.Image, "save", lambda self, file, *args, **kwargs: file.write(bytes(513)))
    with pytest.raises(ValueError, match="cannot fit"):
        avif_codec.encode(face)


def test_avif_preview_qr_uses_cached_complete_file(monkeypatch, tmp_path, face):
    from demo import codec
    monkeypatch.setattr(codec, "DEMO_OUTPUT", tmp_path)
    runtime = DemoRuntime.__new__(DemoRuntime)
    runtime.models = {}
    class Aligned:
        def align(self, image):
            return image
    # Exercise the real four-codec helper with small stand-ins for the neural paths.
    class Neural:
        def __init__(self, size):
            self.size = size
        def encode(self, image):
            return bytes(self.size)
        def reconstruct(self, payload):
            return face
    runtime.models = {n: Neural(n) for n in (64, 128, 256)}
    result = encode_upload(face, Aligned(), runtime, "avif")
    payloads = result[-1]
    monkeypatch.setattr(avif_codec, "encode", lambda *_: pytest.fail("Cached preview was re-encoded"))
    qr, download, status = preview_qr(payloads, "avif", runtime)
    assert "AVIF" in status and read_binary_qr(qr) == payloads["avif"]
    assert result[9] == avif_codec.description(payloads["avif"])
    with Image.open(download) as image:
        assert read_binary_qr(image) == payloads["avif"]
    reconstructed, _, status = decode_upload(qr, runtime)
    np.testing.assert_array_equal(np.asarray(reconstructed), np.asarray(result[5]))
    assert "AVIF" in status
