from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from demo.codec import (
    CHECKPOINT,
    DEFAULT_PAYLOAD_BYTES,
    DemoRuntime,
    Face24Aligner,
    Phase18Runtime,
    Phase22Runtime,
    Phase21Runtime,
    align_from_landmarks,
    decode_upload,
    encode_upload,
    five_points_from_mesh,
    make_binary_qr,
    preview_qr,
    read_binary_qr,
)
from face24.codec64 import decode_bytes


def _test_photo() -> Path:
    """A real single-face photo for end-to-end tests; set FACE24_TEST_PHOTO to enable them."""
    path = os.environ.get("FACE24_TEST_PHOTO")
    if not path or not Path(path).is_file():
        pytest.skip("Set FACE24_TEST_PHOTO to a single-face JPG/PNG to run the real-photo workflow")
    return Path(path)


@pytest.fixture(scope="module")
def runtime():
    return Phase18Runtime(device="cpu")


@pytest.fixture(scope="module")
def demo_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield DemoRuntime(device="cpu")
    finally:
        torch.set_num_threads(previous)


def _fixture_mesh():
    points = np.zeros((478, 2), dtype=np.float64)
    points[:] = [75, 100]
    # Inputs are normalized to the source image dimensions inside the helper.
    for index, point in {
        33: [45, 75], 133: [55, 75], 362: [95, 75], 263: [105, 75],
        1: [75, 108], 61: [60, 133], 291: [90, 133],
        70: [56, 63], 105: [64, 63], 336: [86, 63], 300: [94, 63], 152: [75, 165],
    }.items():
        points[index] = point
    return points


def test_face24_alignment_is_gray_64x80_with_expected_eye_geometry():
    width, height = 178, 218
    mesh = _fixture_mesh()
    normalized = mesh / np.asarray([width, height])
    five = five_points_from_mesh(normalized, width, height)
    np.testing.assert_allclose(five, [[50, 75], [100, 75], [75, 108], [60, 133], [90, 133]])
    image = Image.fromarray(np.random.default_rng(19).integers(0, 256, (height, width, 3), dtype=np.uint8))
    aligned = align_from_landmarks(image, five, mesh)
    assert aligned.mode == "L" and aligned.size == (64, 80)
    pixels = np.asarray(aligned)
    assert pixels.dtype == np.uint8 and pixels.min() >= 0 and pixels.max() <= 255
    from face24.alignment import apply_transform, similarity
    eye_output = apply_transform(five[:2], similarity(five, 24, 23)["matrix_source_to_output"])
    np.testing.assert_allclose(eye_output, [[19.5, 23.0], [43.5, 23.0]], atol=1e-8)


@pytest.mark.parametrize("payload", [bytes(64), bytes(range(64)), bytes([0, 255, 128, 127, 0x80, 0xFF, 1, 0] * 8),
                                    bytes(256), bytes(range(256)), bytes([0, 255, 128, 127, 0x80, 0xFF, 1, 0] * 32),
                                    bytes(128), bytes(range(128)), bytes([0,255,128,1] * 32)])
def test_qr_byte_mode_roundtrip_preserves_arbitrary_binary(payload):
    qr = make_binary_qr(payload)
    assert read_binary_qr(qr) == payload


def test_qr_payload_is_64_row_major_code_indices_and_decodes_without_encoder(runtime):
    payload = bytes(range(64))
    recovered = read_binary_qr(make_binary_qr(payload))
    assert recovered == payload
    assert np.frombuffer(recovered, dtype=np.uint8).reshape(8, 8).tolist() == [list(range(i, i + 8)) for i in range(0, 64, 8)]
    # Decoder-only reconstruction consumes only the bytes recovered from QR.
    face = runtime.reconstruct(recovered)
    assert face.mode == "L" and face.size == (64, 80)
    direct = decode_bytes(recovered, runtime.decoder)
    np.testing.assert_array_equal(np.asarray(face), np.rint(np.clip(direct, 0, 1) * 255).astype(np.uint8))


def test_qr_rejects_invalid_images_and_payload_lengths():
    with pytest.raises(ValueError, match="64, 128, or 256"):
        make_binary_qr(bytes(63))
    with pytest.raises(ValueError, match="64, 128, or 256"):
        make_binary_qr(bytes(65))
    with pytest.raises(ValueError):
        read_binary_qr(Image.new("RGB", (180, 180), "white"))


def test_phase18_encoder_exact_payload_order_and_decoder_export_agreement(runtime):
    image = np.random.default_rng(20).integers(0, 256, (80, 64), dtype=np.uint8)
    payload = runtime.encode(Image.fromarray(image))
    assert len(payload) == 64
    x = torch.from_numpy(image.astype(np.float32)[None, None] / 255).to(runtime.device)
    with torch.inference_mode():
        indices = runtime.encoder.quantizer.indices(runtime.encoder.encoder(x))[0].cpu().numpy().astype(np.uint8)
    assert payload == indices.tobytes(order="C")
    # Runtime construction verifies exact codebook and decoder tensor equality;
    # this checks the callable paths agree on an actual model payload too.
    from face24.codec64 import load_full
    full, _ = load_full(CHECKPOINT, "cpu")
    with torch.inference_mode():
        full_output = full(torch.from_numpy(image.astype(np.float32)[None, None] / 255))[0][0, 0].numpy()
    exported_output = decode_bytes(payload, runtime.decoder)
    np.testing.assert_allclose(full_output, exported_output, rtol=0, atol=0)


def test_complete_upload_face24_qr_scan_reconstruction_workflow(runtime):
    source = Image.open(_test_photo())
    aligner = Face24Aligner()
    try:
        face24 = aligner.align(source)
    finally:
        aligner.close()
        source.close()
    assert face24.mode == "L" and face24.size == (64, 80)
    encoded = runtime.encode(face24)
    assert len(encoded) == 64
    qr_png = make_binary_qr(encoded)
    # This is a real scan of generated QR pixels, never substitution of encoded.
    recovered = read_binary_qr(qr_png)
    assert recovered == encoded
    decoded = runtime.reconstruct(recovered)
    assert decoded.mode == "L" and decoded.size == (64, 80)


@pytest.mark.parametrize("size", [64, 128, 256])
def test_known_formats_dispatch_to_matching_decoder(demo_runtime, size):
    payload = bytes(range(size))
    routed = demo_runtime.reconstruct(payload)
    direct = demo_runtime.models[size].reconstruct(payload)
    np.testing.assert_array_equal(np.asarray(routed), np.asarray(direct))
    assert demo_runtime.models[size].payload_bytes == size
    for invalid in (bytes(63), bytes(65), bytes(95), bytes(96), bytes(97), bytes(127), bytes(129), bytes(255), bytes(257), "x" * size):
        with pytest.raises((TypeError, ValueError)):
            demo_runtime.reconstruct(invalid)
    with pytest.raises(ValueError):
        demo_runtime.models[size].reconstruct(bytes(96 if size == 64 else 64))


def test_phase22_encoder_uses_selected_checkpoint_and_row_major_payload(demo_runtime):
    runtime256 = demo_runtime.models[256]
    pixels = np.random.default_rng(20).integers(0, 256, (80, 64), dtype=np.uint8)
    payload = runtime256.encode(Image.fromarray(pixels))
    x = torch.from_numpy(pixels.astype(np.float32)[None, None] / 255)
    with torch.inference_mode():
        full, indices, _, _ = runtime256.encoder(x)
    assert len(payload) == 256
    assert payload == indices[0].numpy().astype(np.uint8).tobytes(order="C")
    from face24.codec256 import decode_bytes as decode256
    np.testing.assert_allclose(full[0, 0].numpy(), decode256(payload, runtime256.decoder), rtol=0, atol=2e-6)
    assert indices.shape == (1, 2, 16, 8)
    assert payload[:128] == indices[0, 0].numpy().astype(np.uint8).tobytes(order="C")
    assert payload[128:] == indices[0, 1].numpy().astype(np.uint8).tobytes(order="C")


@pytest.mark.parametrize("size", [64, 128, 256])
def test_photo_comparison_and_qr_export_use_selected_model(demo_runtime, monkeypatch, tmp_path, size):
    import json

    from demo import codec

    monkeypatch.setattr(codec, "DEMO_OUTPUT", tmp_path)
    photo = _test_photo()
    aligner = Face24Aligner()
    try:
        with Image.open(photo) as source:
            if size == DEFAULT_PAYLOAD_BYTES:
                results = encode_upload(source, aligner, demo_runtime)
            else:
                results = encode_upload(source, aligner, demo_runtime, size)
    finally:
        aligner.close()
    _original, aligned, face64, face128, face256, avif_preview, qr, download, status, avif_note, avif_file, payloads = results
    assert avif_preview.size == (64, 80) and "overhead" in avif_note
    assert Path(avif_file).read_bytes() == payloads["avif"]
    previews = {64: face64, 128: face128, 256: face256}
    phase = {64: 18, 128: 21, 256: 22}[size]
    payload = read_binary_qr(qr)
    assert len(payload) == size
    assert payload == payloads[size]
    assert f"{size} bytes" in status
    assert f"Phase {phase}" in status
    for length, preview in previews.items():
        model = demo_runtime.models[length]
        expected = model.reconstruct(model.encode(aligned))
        np.testing.assert_array_equal(np.asarray(preview), np.asarray(expected))
    with Image.open(download) as saved_qr:
        assert read_binary_qr(saved_qr) == payload
    decoded, decoded_download, decoded_status = decode_upload(qr, demo_runtime)
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(previews[size]))
    assert Path(decoded_download).is_file()
    assert f"Phase {phase}" in decoded_status


def test_phase22_rejects_changed_checkpoint_before_loading(monkeypatch, tmp_path):
    import json

    from demo import codec

    completion = json.loads((codec.PHASE22 / "training_completion.json").read_text(encoding="utf-8"))
    (tmp_path / "training_completion.json").write_text(json.dumps(completion), encoding="utf-8")
    (tmp_path / "best.pt").write_bytes(b"incorrect checkpoint")
    monkeypatch.setattr(Phase22Runtime, "directory", tmp_path)
    with pytest.raises(RuntimeError, match="Phase 22 RVQ selected checkpoint hash mismatch"):
        Phase22Runtime(device="cpu")


@pytest.mark.parametrize("size", [63, 65, 95, 96, 97, 127, 129, 255, 257])
def test_scanner_rejects_other_lengths_even_for_valid_qr_images(size):
    import qrcode
    from qrcode.util import MODE_8BIT_BYTE, QRData

    qr = qrcode.QRCode(box_size=8, border=4)
    qr.add_data(QRData(bytes(i % 256 for i in range(size)), mode=MODE_8BIT_BYTE), optimize=0)
    qr.make(fit=True)
    with pytest.raises(ValueError, match="expected 64, 128, or 256"):
        read_binary_qr(qr.make_image().convert("RGB"))


def test_phase21_encoder_and_export_agree_on_selected_model(demo_runtime):
    from face24.codec128 import decode_bytes as decode128

    runtime128 = demo_runtime.models[128]
    pixels = np.random.default_rng(21).integers(0, 256, (80, 64), dtype=np.uint8)
    payload = runtime128.encode(Image.fromarray(pixels))
    with torch.inference_mode():
        full, indices, _, _ = runtime128.encoder(torch.from_numpy(pixels.astype(np.float32)[None, None] / 255))
    assert len(payload) == 128
    assert payload == indices[0].numpy().astype(np.uint8).tobytes(order="C")
    np.testing.assert_array_equal(full[0, 0].numpy(), decode128(payload, runtime128.decoder))


def test_preview_click_exports_cached_payload_without_another_encoding(demo_runtime, monkeypatch, tmp_path):
    from demo import codec

    monkeypatch.setattr(codec, 'DEMO_OUTPUT', tmp_path)
    pixels = Image.fromarray(np.random.default_rng(22).integers(0, 256, (80, 64), dtype=np.uint8))
    class AlreadyAligned:
        def align(self, image):
            return image
    results = encode_upload(pixels, AlreadyAligned(), demo_runtime)
    previews = dict(zip((64, 128, 256), results[2:5]))
    payloads = results[-1]
    def unexpected_encode(*args):
        raise AssertionError('Preview clicks must not encode any image again')
    for model in demo_runtime.models.values():
        monkeypatch.setattr(model, 'encode', unexpected_encode)
    for length in (64, 128, 256):
        qr, download, status = preview_qr(payloads, length, demo_runtime)
        recovered = read_binary_qr(qr)
        assert recovered == payloads[length]
        np.testing.assert_array_equal(np.asarray(demo_runtime.reconstruct(recovered)), np.asarray(previews[length]))
        assert Path(download).is_file() and f'{length} bytes' in status
    for invalid_state in ({}, None, {128: bytes(64)}):
        with pytest.raises(ValueError, match='Encode a photograph first'):
            preview_qr(invalid_state, 128, demo_runtime)


def test_phase21_rejects_changed_checkpoint_before_loading(monkeypatch, tmp_path):
    import json

    from demo import codec

    completion = json.loads((codec.PHASE21 / 'training_completion.json').read_text(encoding='utf-8'))
    (tmp_path/'training_completion.json').write_text(json.dumps(completion), encoding='utf-8')
    (tmp_path/'best.pt').write_bytes(b'wrong phase21 checkpoint')
    monkeypatch.setattr(Phase21Runtime, 'directory', tmp_path)
    with pytest.raises(RuntimeError, match='Phase 21 selected checkpoint hash mismatch'):
        Phase21Runtime(device='cpu')

