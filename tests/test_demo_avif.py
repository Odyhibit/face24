import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from demo import avif_codec
from demo.codec import DemoRuntime, encode_upload, preview_qr, read_binary_qr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def face():
    # Structured gradients with edges exercise real image data at low bitrates.
    y, x = np.indices((80, 64))
    return Image.fromarray(((x * 3 + y * 2 + (x > 32) * 50) % 256).astype(np.uint8))


def test_template_matches_the_browser_demo():
    source = (ROOT / "web" / "avif.js").read_text(encoding="utf-8")
    web = base64.b64decode(re.search(r'TEMPLATE_B64 =\s*"([^"]+)"', source).group(1))
    assert avif_codec.TEMPLATE == web


def test_matches_browser_reference_vectors():
    """Same frames, rebuilt files and decoded pixels as web/avif_vectors.json (made with Pillow)."""
    vectors = json.loads((ROOT / "web" / "avif_vectors.json").read_text(encoding="utf-8"))
    b64 = base64.b64decode
    assert vectors["cases"]
    for case in vectors["cases"]:
        image = Image.frombytes("L", (64, 80), b64(vectors["inputs"][case["input"]]))
        payload = avif_codec.encode_frame(image, case["quality"])
        assert payload == b64(case["payload"])
        assert avif_codec.rebuild(payload) == b64(case["rebuilt"])
        assert np.asarray(avif_codec.decode(payload)).tobytes() == b64(case["decoded"])


def test_strip_and_rebuild_round_trip(face):
    buffer = io.BytesIO()
    face.save(buffer, "AVIF", quality=40, speed=4, subsampling="4:0:0", max_threads=1)
    full = buffer.getvalue()
    payload = avif_codec.strip(full)
    assert len(payload) < len(full) - 250  # the template carries the ~280 fixed bytes
    assert avif_codec.rebuild(payload) == full
    with Image.open(io.BytesIO(full)) as image:
        expected = np.asarray(image.convert("L"))
    np.testing.assert_array_equal(np.asarray(avif_codec.decode(payload)), expected)


def test_each_budget_gets_the_smallest_frame_at_least_that_large(face):
    found = avif_codec.encode_for_budgets(face)
    assert set(found) == set(avif_codec.BUDGETS)
    # Reproduce the scan: qualities 0..100, stopping after the first frame past 2x the largest budget.
    sizes, stop = [], 2 * max(avif_codec.BUDGETS)
    for quality in avif_codec.QUALITIES:
        sizes.append(len(avif_codec.encode_frame(face, quality)))
        if sizes[-1] > stop:
            break
    for budget, entry in found.items():
        assert len(entry["payload"]) >= budget
        assert entry["payload"] == avif_codec.encode_frame(face, entry["quality"])
        assert len(entry["payload"]) == min(size for size in sizes if size >= budget)
    assert "AVIF" in avif_codec.description(found)
    assert str(len(found[128]["payload"])) in avif_codec.caption(found, 128)


def test_scan_prefers_higher_quality_on_ties_and_skips_unreachable_budgets(monkeypatch, face):
    sizes = {0: 70, 1: 66, 2: 66, 3: 90}
    monkeypatch.setattr(avif_codec, "encode_frame", lambda image, quality: bytes(sizes.get(quality, 600)))
    found = avif_codec.encode_for_budgets(face, budgets=(64, 1000))
    assert found[64]["quality"] == 2 and len(found[64]["payload"]) == 66
    assert 1000 not in found
    assert "no frame" in avif_codec.caption(found, 1000)


@pytest.mark.parametrize("payload", [b"", b"\xff" * 5])
def test_corrupt_frames_are_rejected(payload):
    with pytest.raises(ValueError):
        avif_codec.decode(payload)


def test_avif_is_comparison_only(monkeypatch, tmp_path, face):
    from demo import codec
    monkeypatch.setattr(codec, "DEMO_OUTPUT", tmp_path)
    class Aligned:
        def align(self, image):
            return image
    # Small stand-ins for the neural paths; the AVIF path is real.
    class Neural:
        def __init__(self, size):
            self.size = size
            self.label = f"Stand-in {size}"
        def encode(self, image):
            return bytes(self.size)
        def reconstruct(self, payload):
            return face
    runtime = DemoRuntime.__new__(DemoRuntime)
    runtime.models = {n: Neural(n) for n in (64, 128, 256)}
    result = encode_upload(face, Aligned(), runtime)
    payloads = result[-1]
    assert set(payloads) == {64, 128, 256}
    for avif_preview in result[3:8:2]:
        assert avif_preview.size == (64, 80)
    with pytest.raises(ValueError):
        preview_qr(payloads, "avif", runtime)
    qr, _, status = preview_qr(payloads, 128, runtime)
    assert read_binary_qr(qr) == payloads[128] and "128 bytes" in status
