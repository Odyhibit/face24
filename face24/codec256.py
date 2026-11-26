"""256-byte codec (Phase 22): two-stage residual VQ over the 128-byte architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .codec128 import Decoder128, Face24VQVAE128


@dataclass(frozen=True)
class RVQConfig:
    architecture: str = "face24-rvq256-v1"
    image_size: tuple[int, int] = (80, 64)
    grid: tuple[int, int] = (16, 8)
    stages: int = 2
    entries: int = 256
    embedding_dim: int = 64
    payload_bytes: int = 256
    index_order: str = "stage-major-row-major-yx"

    def __post_init__(self):
        if (
            self.architecture != "face24-rvq256-v1"
            or tuple(self.image_size) != (80, 64)
            or tuple(self.grid) != (16, 8)
            or self.stages != 2
            or self.entries != 256
            or self.embedding_dim != 64
            or self.payload_bytes != 256
            or self.index_order != "stage-major-row-major-yx"
        ):
            raise ValueError("Unsupported Phase22 RVQ configuration")


def nearest(features, embedding):
    with (
        torch.no_grad(),
        torch.autocast(device_type=features.device.type, enabled=False),
    ):
        vectors = features.float().permute(0, 2, 3, 1).reshape(-1, 64)
        weights = embedding.weight.float()
        distances = (
            vectors.square().sum(1, keepdim=True)
            + weights.square().sum(1)[None]
            - 2 * vectors @ weights.T
        )
        return distances.argmin(1).reshape(-1, 16, 8)


class ResidualQuantizer(nn.Module):
    def __init__(self, first_embedding=None):
        super().__init__()
        self.embedding1 = (
            first_embedding if first_embedding is not None else nn.Embedding(256, 64)
        )
        if first_embedding is None:
            nn.init.uniform_(self.embedding1.weight, -1 / 256, 1 / 256)
        self.embedding2 = nn.Embedding(256, 64)
        nn.init.uniform_(self.embedding2.weight, -1 / 256, 1 / 256)
        self.last_diagnostics = {}

    def details(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 16, 8):
            raise ValueError("Expected N x 64 x 16 x 8 continuous features")
        z = z.float()
        i1 = nearest(z, self.embedding1)
        q1 = self.embedding1(i1).permute(0, 3, 1, 2).contiguous().float()
        residual = z - q1.detach()
        i2 = nearest(residual, self.embedding2)
        q2 = self.embedding2(i2).permute(0, 3, 1, 2).contiguous().float()
        cb1, cb2 = F.mse_loss(q1, z.detach()), F.mse_loss(q2, residual.detach())
        cm1, cm2 = F.mse_loss(z, q1.detach()), F.mse_loss(residual, q2.detach())
        # Exactly one identity-Jacobian straight-through path to the encoder.
        combined = (q1 + q2).detach() + (z - z.detach())
        return {
            "z": z,
            "q1": q1,
            "q2": q2,
            "residual": residual,
            "indices": torch.stack((i1, i2), dim=1),
            "quantized": combined,
            "codebook1": cb1,
            "codebook2": cb2,
            "commitment1": cm1,
            "commitment2": cm2,
            "codebook": (cb1 + cb2) / 2,
            "commitment": (cm1 + cm2) / 2,
        }

    @torch.no_grad()
    def indices(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 16, 8):
            raise ValueError("Expected N x 64 x 16 x 8 continuous features")
        i1 = nearest(z, self.embedding1)
        q1 = self.embedding1(i1).permute(0, 3, 1, 2).contiguous()
        i2 = nearest(z.float() - q1.float(), self.embedding2)
        return torch.stack((i1, i2), dim=1)

    def forward(self, z):
        d = self.details(z)
        self.last_diagnostics = {
            key: d[key].detach()
            for key in ("codebook1", "codebook2", "commitment1", "commitment2")
        }
        self.last_diagnostics.update(
            {
                "feature_error_before_stage1": d["z"].detach().square().mean(),
                "feature_error_after_stage1": d["residual"].detach().square().mean(),
                "feature_error_after_stage2": (d["residual"] - d["q2"])
                .detach()
                .square()
                .mean(),
                "q1_norm_mean": d["q1"].detach().norm(dim=1).mean(),
                "q2_norm_mean": d["q2"].detach().norm(dim=1).mean(),
                "qsum_norm_mean": (d["q1"] + d["q2"]).detach().norm(dim=1).mean(),
            }
        )
        return d["quantized"], d["indices"], d["codebook"], d["commitment"]


class Face24RVQVAE(nn.Module):
    def __init__(self, config=RVQConfig()):
        super().__init__()
        self.config = config
        # A fresh Phase21 architecture preserves its seeded encoder/decoder
        # initialization. No trained Phase21 weights are loaded.
        scratch = Face24VQVAE128()
        self.encoder = scratch.encoder
        self.quantizer = ResidualQuantizer(scratch.quantizer.embedding)
        self.decoder = scratch.decoder

    def forward(self, images):
        quantized, indices, codebook, commitment = self.quantizer(self.encoder(images))
        return self.decoder(quantized), indices, codebook, commitment


class DecoderOnlyRVQ(nn.Module):
    def __init__(self, config=RVQConfig()):
        super().__init__()
        self.config = config
        self.codebook1 = nn.Embedding(256, 64)
        self.codebook2 = nn.Embedding(256, 64)
        self.decoder = Decoder128()
        self.requires_grad_(False)

    def forward(self, indices, coarse_only=False):
        validate_indices(indices)
        q = self.codebook1(indices[:, 0])
        if not coarse_only:
            q = q + self.codebook2(indices[:, 1])
        return self.decoder(q.permute(0, 3, 1, 2).contiguous())


def validate_indices(indices):
    if (
        indices.ndim != 4
        or tuple(indices.shape[1:]) != (2, 16, 8)
        or indices.dtype != torch.long
        or indices.min() < 0
        or indices.max() > 255
    ):
        raise ValueError("Expected N x 2 x 16 x 8 long indices in [0,255]")


def serialize_indices(indices):
    value = torch.as_tensor(indices)
    if (
        tuple(value.shape) != (2, 16, 8)
        or value.dtype
        not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        or value.min() < 0
        or value.max() > 255
    ):
        raise ValueError("Expected 2 x 16 x 8 integer indices in [0,255]")
    return value.to(torch.uint8).detach().cpu().numpy().tobytes(order="C")


def decode_indices(payload):
    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 256:
        raise ValueError(
            "Phase22 requires exactly 256 raw bytes, including both stages"
        )
    return torch.from_numpy(
        np.frombuffer(payload, np.uint8).copy().reshape(1, 2, 16, 8).astype(np.int64)
    )


@torch.inference_mode()
def encode_array(values, model):
    pixels = np.asarray(values, np.float32)
    if (
        pixels.size != 5120
        or not np.isfinite(pixels).all()
        or pixels.min() < 0
        or pixels.max() > 1
    ):
        raise ValueError("Expected finite normalized 80x64 grayscale pixels")
    model.eval()
    x = torch.from_numpy(pixels.reshape(1, 1, 80, 64).copy()).to(
        next(model.parameters()).device
    )
    return serialize_indices(model.quantizer.indices(model.encoder(x))[0])


def decode_bytes(payload, decoder, coarse_only=False):
    indices = decode_indices(payload).to(next(decoder.parameters()).device)
    decoder.eval()
    with torch.inference_mode():
        return decoder(indices, coarse_only=coarse_only)[0, 0].cpu().numpy()


def parameter_counts(model):
    return {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "codebook1": model.quantizer.embedding1.weight.numel(),
        "codebook2": model.quantizer.embedding2.weight.numel(),
        "codebook": sum(p.numel() for p in model.quantizer.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "full": sum(p.numel() for p in model.parameters()),
    }


def _load(path, kind, device):
    bundle = torch.load(
        Path(path), map_location="cpu", weights_only=kind == "rvqdecoder"
    )
    if (
        bundle.get("format_version") != 1
        or bundle.get("codec") != "face24-rvq256-v1"
        or bundle.get("kind") != kind
    ):
        raise ValueError("Unsupported Phase22 RVQ model artifact")
    config = RVQConfig(**bundle["config"])
    model = DecoderOnlyRVQ(config) if kind == "rvqdecoder" else Face24RVQVAE(config)
    model.load_state_dict(bundle["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite RVQ weights")
    return model.to(device).eval(), bundle


def load_full(path, device="cpu"):
    return _load(path, "rvqvae", device)


def load_decoder(path, device="cpu"):
    return _load(path, "rvqdecoder", device)
