"""Inference-only Face24 codecs and binary QR helpers for the local demo."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from starlette.middleware import Middleware

from demo import avif_codec
from demo.frontend import FrontendMetadataMiddleware
from face24 import codec128 as face24_128_model, codec256 as face24_rvq_model
from face24.codec64 import (
    decode_bytes,
    encode_array,
    load_decoder,
    load_full,
    parameter_counts,
)
from face24.alignment import apply_transform, render_crop, similarity

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
PHASE18 = WEIGHTS / "face24_64"
CHECKPOINT = PHASE18 / "best.pt"
DECODER_EXPORT = PHASE18 / "decoder.pt"
COMPLETION = PHASE18 / "training_completion.json"
PHASE21 = WEIGHTS / "face24_128"
PHASE22 = WEIGHTS / "face24_256_rvq"
DEFAULT_PAYLOAD_BYTES = 128
PAYLOAD_LENGTHS = (64, 128, 256)
MODEL_CHOICES = [("Phase 18 · 64 bytes", 64), ("Phase 21 · 128 bytes (preferred)", 128),
                 ("Phase 22 RVQ · 256 bytes", 256), ("AVIF · up to 512 bytes", "avif")]
# MediaPipe Face Landmarker v1 (float16), Apache-2.0:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
DETECTOR_MODEL = WEIGHTS / "mediapipe/face_landmarker.task"
DETECTOR_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEMO_OUTPUT = ROOT / "output"
EXPECTED_ARC = 24.0
EXPECTED_EYE = (31.5, 23.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def five_points_from_mesh(mesh, width: int, height: int) -> np.ndarray:
    """Return screen-left/right eye centers, nose tip, and screen-left/right mouth corners."""
    try:
        p = np.asarray(mesh, dtype=np.float64)
    except (TypeError, ValueError):
        try:
            p = np.asarray([[point.x, point.y] for point in mesh], dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.") from exc
    if p.ndim != 2 or p.shape[0] < 468 or p.shape[1] < 2 or not np.isfinite(p[:, :2]).all():
        raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.")
    xy = p[:, :2] * np.asarray([width, height], dtype=np.float64)
    # MediaPipe Face Mesh's eye corner pairs; centers match the Phase 10C
    # five-point eye-center convention. Sort by image x for stable orientation.
    eyes = [xy[[33, 133]].mean(axis=0), xy[[362, 263]].mean(axis=0)]
    eyes.sort(key=lambda q: q[0])
    mouth = [xy[61], xy[291]]
    mouth.sort(key=lambda q: q[0])
    points = np.asarray([eyes[0], eyes[1], xy[1], mouth[0], mouth[1]], dtype=np.float64)
    iod = float(np.linalg.norm(points[1] - points[0]))
    if points[1, 0] <= points[0, 0] or not 2.0 <= iod <= min(width, height) * 0.8:
        raise ValueError("Detected eye landmarks have implausible geometry; try a clearer, front-facing photo.")
    return points


def _verify_detector_asset() -> None:
    if not DETECTOR_MODEL.is_file() or _sha256(DETECTOR_MODEL) != DETECTOR_SHA256:
        raise RuntimeError("The local MediaPipe face-landmark model is missing or failed its recorded hash check.")


def align_from_landmarks(image: Image.Image, five_points, dense_points) -> Image.Image:
    """Apply the frozen Face24 B geometry to pixel-space five/dense landmarks."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    five = np.asarray(five_points, dtype=np.float64)
    dense = np.asarray(dense_points, dtype=np.float64)
    if five.shape != (5, 2) or dense.ndim != 2 or dense.shape[0] < 468 or dense.shape[1] != 2:
        raise ValueError("Face landmarks are missing or invalid; try a clearer, front-facing photo.")
    if not np.isfinite(five).all() or not np.isfinite(dense).all():
        raise ValueError("Face landmarks contain invalid coordinates.")
    transform = similarity(five, EXPECTED_ARC, EXPECTED_EYE[1])
    matrix = np.asarray(transform["matrix_source_to_output"], dtype=np.float64)
    important = np.vstack([five, dense[[70, 105, 336, 300, 152]]])
    fitted = apply_transform(important, matrix)
    if not np.isfinite(fitted).all() or np.any(fitted[:, 0] < -0.5) or np.any(fitted[:, 0] > 63.5) or np.any(fitted[:, 1] < -0.5) or np.any(fitted[:, 1] > 79.5):
        raise ValueError("This face's critical landmarks do not fit the Face24 crop. Try a more frontal or less tightly framed photo.")
    gray, _, _ = render_crop(image, np.ones((h, w), dtype=np.float32), transform)
    if gray.mode != "L" or gray.size != (64, 80):
        raise RuntimeError("Face24 renderer returned an invalid image format.")
    return gray


class Face24Aligner:
    """One persistent MediaPipe detector and the published Face24 renderer."""

    def __init__(self):
        _verify_detector_asset()
        import mediapipe as mp

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(DETECTOR_MODEL)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=8,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self._mp = mp
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._detector.close()

    def align(self, image: Image.Image) -> Image.Image:
        if image is None:
            raise ValueError("Upload a JPG or PNG face photograph first.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        frame = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._detector.detect(frame)
        count = len(result.face_landmarks)
        if count == 0:
            raise ValueError("No face was detected. Try a clear, well-lit image with the face visible.")
        if count != 1:
            raise ValueError(f"Detected {count} faces. Please provide an image containing exactly one face.")
        h, w = rgb.shape[:2]
        five = five_points_from_mesh(result.face_landmarks[0], w, h)
        mesh = np.asarray([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float64)
        return align_from_landmarks(image, five, mesh)


class _PhaseRuntime:
    """Loads and verifies the matching encoder and CPU decoder exactly once."""

    expected_counts = {"encoder": 517408, "codebook": 16384, "decoder": 3411585, "full": 3945377}
    codebook_keys = (("quantizer.embedding.weight", "codebook.weight"),)

    def __init__(self, device: str | None = None):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        completion_path = self.directory / "training_completion.json"
        checkpoint = self.directory / "best.pt"
        decoder_export = self.directory / "decoder.pt"
        if not completion_path.is_file():
            raise RuntimeError(f"{self.label} training completion record is missing.")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("selected_epoch") != self.selected_epoch or completion.get("optimizer_steps") != 58000:
            raise RuntimeError(f"{self.label} selected-checkpoint provenance is invalid.")
        if _sha256(checkpoint) != completion["best_checkpoint_sha256"]:
            raise RuntimeError(f"{self.label} selected checkpoint hash mismatch.")
        if _sha256(decoder_export) != completion["decoder_sha256"]:
            raise RuntimeError(f"{self.label} decoder export hash mismatch.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, bundle = self.load_full(checkpoint, self.device)
        self.decoder, decoder_bundle = self.load_decoder(decoder_export, "cpu")
        if (bundle.get("codec") != self.codec or bundle.get("kind") != self.kind
                or bundle.get("checkpoint_role") != "selection" or bundle.get("epoch") != completion["selected_epoch"]
                or bundle.get("experiment_sha256") != completion["experiment_sha256"]):
            raise RuntimeError(f"Loaded checkpoint is not the selected {self.label} model.")
        if self.parameter_counts(self.encoder) != self.expected_counts:
            raise RuntimeError(f"{self.label} architecture does not match the frozen codec specification.")
        full_state = bundle["state_dict"]
        exported_state = decoder_bundle["state_dict"]
        for full_key, export_key in self.codebook_keys:
            if not torch.equal(full_state[full_key].cpu(), exported_state[export_key].cpu()):
                raise RuntimeError(f"{self.label} decoder export uses a different codebook.")
        for key, value in self.encoder.decoder.state_dict().items():
            if not torch.equal(value.cpu(), exported_state["decoder." + key].cpu()):
                raise RuntimeError(f"{self.label} decoder export does not match the selected checkpoint.")
        self._lock = threading.Lock()

    def encode(self, face24: Image.Image) -> bytes:
        gray = np.asarray(face24.convert("L"), dtype=np.uint8)
        if gray.shape != (80, 64):
            raise ValueError("Expected a 64×80 Face24 image.")
        with self._lock, torch.inference_mode():
            payload = self.encode_array(gray.astype(np.float32) / 255.0, self.encoder)
        if len(payload) != self.payload_bytes:
            raise RuntimeError(f"Encoder returned {len(payload)} bytes; expected exactly {self.payload_bytes}.")
        return payload

    def reconstruct(self, payload: bytes) -> Image.Image:
        if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != self.payload_bytes:
            raise ValueError(f"Invalid {self.label} payload: expected exactly {self.payload_bytes} bytes.")
        with self._lock:
            pixels = self.decode_bytes(payload, self.decoder)
        values = np.asarray(pixels, dtype=np.float32)
        if values.shape != (80, 64) or not np.isfinite(values).all():
            raise ValueError(f"{self.label} decoder returned invalid pixels.")
        return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))


class Phase18Runtime(_PhaseRuntime):
    label = "Phase 18"
    directory = PHASE18
    selected_epoch = 147
    payload_bytes = 64
    codec = "vqface-face24-large-v1"
    kind = "vqvae"
    load_full = staticmethod(load_full)
    load_decoder = staticmethod(load_decoder)
    parameter_counts = staticmethod(parameter_counts)
    encode_array = staticmethod(encode_array)
    decode_bytes = staticmethod(decode_bytes)


class DemoRuntime:
    """The three known formats; dispatch is entirely determined by payload length."""

    def __init__(self, device: str | None = None):
        self.models = {64: Phase18Runtime(device), 128: Phase21Runtime(device), 256: Phase22Runtime(device)}

    def model_for_length(self, payload_bytes: int) -> _PhaseRuntime:
        try:
            return self.models[payload_bytes]
        except (KeyError, TypeError) as exc:
            raise ValueError("Select Phase 18 (64 bytes), Phase 21 (128 bytes), or Phase 22 RVQ (256 bytes).") from exc

    def reconstruct(self, payload: bytes) -> Image.Image:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("Expected a raw binary face payload.")
        if avif_codec.is_avif(payload):
            return avif_codec.decode(bytes(payload))
        return self.model_for_length(len(payload)).reconstruct(payload)


class Phase21Runtime(_PhaseRuntime):
    label = "Phase 21"
    directory = PHASE21
    selected_epoch = 187
    payload_bytes = 128
    codec = "vqface-face24-128-v1"
    kind = "vqvae128"
    load_full = staticmethod(face24_128_model.load_full)
    load_decoder = staticmethod(face24_128_model.load_decoder)
    parameter_counts = staticmethod(face24_128_model.parameter_counts)
    encode_array = staticmethod(face24_128_model.encode_array)
    decode_bytes = staticmethod(face24_128_model.decode_bytes)


class Phase22Runtime(_PhaseRuntime):
    label = "Phase 22 RVQ"
    directory = PHASE22
    selected_epoch = 197
    payload_bytes = 256
    codec = "face24-rvq256-v1"
    kind = "rvqvae"
    expected_counts = {"encoder": 517408, "codebook1": 16384, "codebook2": 16384,
                       "codebook": 32768, "decoder": 3411585, "full": 3961761}
    codebook_keys = (("quantizer.embedding1.weight", "codebook1.weight"),
                     ("quantizer.embedding2.weight", "codebook2.weight"))
    load_full = staticmethod(face24_rvq_model.load_full)
    load_decoder = staticmethod(face24_rvq_model.load_decoder)
    parameter_counts = staticmethod(face24_rvq_model.parameter_counts)
    encode_array = staticmethod(face24_rvq_model.encode_array)
    decode_bytes = staticmethod(face24_rvq_model.decode_bytes)


def make_binary_qr(payload: bytes) -> Image.Image:
    """Create a standard QR with raw byte-mode data, never text/base64/hex."""
    _validate_payload(payload)
    import qrcode
    from qrcode.constants import ERROR_CORRECT_M
    from qrcode.util import MODE_8BIT_BYTE, QRData

    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M, box_size=8, border=4)
    raw = bytes(payload)
    qr.add_data(QRData(raw, mode=MODE_8BIT_BYTE), optimize=0)
    try:
        qr.make(fit=True)
    except ValueError as exc:
        if str(exc) != "glog(0)":
            raise
        # qrcode 8.2's polynomial division rejects an all-zero RS data block.
        # Short byte-mode segments place standard QR mode/count bits between
        # zero runs. Scanners concatenate their bytes into the original payload.
        qr.clear()
        for start in range(0, len(raw), 16):
            qr.add_data(QRData(raw[start:start + 16], mode=MODE_8BIT_BYTE), optimize=0)
        qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def read_binary_qr(image: Image.Image) -> bytes:
    """Decode exactly one QR barcode and return its raw bytes property."""
    import zxingcpp

    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    results = zxingcpp.read_barcodes(bgr)
    if len(results) != 1:
        raise ValueError("No QR code was decoded (or more than one code was found). Upload one clear QR image.")
    result = results[0]
    if str(result.format).lower().split(".")[-1] != "qrcode":
        raise ValueError("The uploaded image is not a QR code.")
    raw = getattr(result, "bytes", None)
    if raw is None:
        raise RuntimeError("The installed QR scanner does not expose raw binary payload bytes.")
    payload = bytes(raw)
    _validate_payload(payload)
    return payload


def _validate_payload(payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("QR payload must contain exactly 64, 128, or 256 raw bytes, or a complete AVIF file up to 512 bytes.")
    if avif_codec.is_avif(payload):
        avif_codec.decode(bytes(payload))
    elif len(payload) not in PAYLOAD_LENGTHS:
        raise ValueError(f"Decoded QR payload has {len(payload)} bytes; expected 64, 128, or 256 bytes, or a complete AVIF file up to 512 bytes.")


def save_demo_image(image: Image.Image, suffix: str) -> str:
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DEMO_OUTPUT / f"{secrets.token_hex(12)}{suffix}"
    image.save(path)
    return str(path)


def encode_upload(image: Image.Image, aligner: Face24Aligner, runtime: DemoRuntime,
                  payload_bytes: int | str = DEFAULT_PAYLOAD_BYTES):
    if payload_bytes != "avif":
        runtime.model_for_length(payload_bytes)
    aligned = aligner.align(image)
    payloads = {length: model.encode(aligned) for length, model in runtime.models.items()}
    payloads["avif"] = avif_codec.encode(aligned)
    comparisons = {length: runtime.reconstruct(payload) for length, payload in payloads.items()}
    qr, download, status = preview_qr(payloads, payload_bytes, runtime)
    return (ImageOps.exif_transpose(image).convert("RGB"), aligned,
            comparisons[64], comparisons[128], comparisons[256], comparisons["avif"],
            qr, download, status, avif_codec.description(payloads["avif"]),
            save_demo_bytes(payloads["avif"], ".avif"), payloads)


def save_demo_bytes(payload: bytes, suffix: str) -> str:
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = DEMO_OUTPUT / f"{secrets.token_hex(12)}{suffix}"
    path.write_bytes(payload)
    return str(path)


def preview_qr(payloads: dict, payload_bytes: int | str, runtime: DemoRuntime):
    """Export the original encoded bytes for a preview, without re-encoding it."""
    selected = runtime.model_for_length(payload_bytes) if payload_bytes != "avif" else None
    payload = payloads.get(payload_bytes) if isinstance(payloads, dict) else None
    if not isinstance(payload, bytes) or (payload_bytes != "avif" and len(payload) != payload_bytes):
        raise ValueError("Encode a photograph first, then click a reconstruction to see its QR code.")
    qr = make_binary_qr(payload)
    # Test the actual generated QR pixels before exposing the result.
    recovered = read_binary_qr(qr)
    if recovered != payload:
        raise RuntimeError(f"QR scanner did not recover the exact {payload_bytes} encoded bytes.")
    download = save_demo_image(qr, ".png")
    label = selected.label if selected else "AVIF · complete file"
    return qr, download, f"{label} · {len(payload)} bytes · QR binary round-trip verified."


def decode_upload(image: Image.Image, runtime: DemoRuntime):
    if image is None:
        raise ValueError("Upload a QR code image first.")
    payload = read_binary_qr(image)
    reconstructed = runtime.reconstruct(payload)
    download = save_demo_image(reconstructed, ".png")
    label = "AVIF · complete file" if avif_codec.is_avif(payload) else runtime.model_for_length(len(payload)).label
    return reconstructed, download, f"Decoded with {label} · {len(payload)} raw bytes."


def create_app():
    from demo.interface import build_app

    return build_app(DemoRuntime(), Face24Aligner())


def main():
    app = create_app()
    # Loopback by default; hosted deployments (e.g. Hugging Face Spaces) set GRADIO_SERVER_NAME=0.0.0.0.
    app.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
               server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), share=False, pwa=False,
               app_kwargs={"middleware": [Middleware(FrontendMetadataMiddleware)]})


if __name__ == "__main__":
    main()
