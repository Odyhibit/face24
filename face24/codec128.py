"""128-byte codec (Phase 21): Face24 VQ-VAE with a native 16x8 latent grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import Residual


@dataclass(frozen=True)
class VQ128Config:
    architecture: str = "vqface-face24-128-v1"
    image_size: tuple[int, int] = (80, 64)
    grid: tuple[int, int] = (16, 8)
    entries: int = 256
    embedding_dim: int = 64
    payload_bytes: int = 128
    index_order: str = "row-major-yx"

    def __post_init__(self):
        if (
            self.architecture != "vqface-face24-128-v1"
            or tuple(self.image_size) != (80, 64)
            or tuple(self.grid) != (16, 8)
            or self.entries != 256
            or self.embedding_dim != 64
            or self.payload_bytes != 128
            or self.index_order != "row-major-yx"
        ):
            raise ValueError("Unsupported Phase21 codec configuration")


class Encoder128(nn.Module):
    """Phase20 learned layers with continuous-feature resampling before quantization."""

    def __init__(self, config: VQ128Config = VQ128Config()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            Residual(128),
            Residual(128),
            nn.ReLU(),
            # Resize continuous 128-channel features, never quantized indices.
            # Padding four rows on the original 10x8 map would yield four
            # bias-only output rows; padding one on this 16x8 map does not.
            nn.Upsample(size=(16, 8), mode="bilinear", align_corners=False),
            nn.Conv2d(128, config.embedding_dim, (3, 1), padding=(1, 0)),
        )

    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (1, 80, 64):
            raise ValueError("Expected N x 1 x 80 x 64 Face24 images")
        value = self.layers(image)
        if tuple(value.shape[1:]) != (64, 16, 8):
            raise RuntimeError(
                f"Phase21 encoder produced {tuple(value.shape[1:])}, expected 64x16x8"
            )
        return value


class VectorQuantizer128(nn.Module):
    def __init__(self, config: VQ128Config = VQ128Config()):
        super().__init__()
        self.embedding = nn.Embedding(config.entries, config.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / config.entries, 1 / config.entries)

    @torch.no_grad()
    def indices(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 16, 8):
            raise ValueError("Expected N x 64 x 16 x 8 features")
        with torch.autocast(device_type=z.device.type, enabled=False):
            vectors = z.float().permute(0, 2, 3, 1).reshape(-1, 64)
            weights = self.embedding.weight.float()
            distances = (
                vectors.square().sum(1, keepdim=True)
                + weights.square().sum(1)[None]
                - 2 * vectors @ weights.T
            )
            return distances.argmin(1).reshape(-1, 16, 8)

    def forward(self, z):
        indices = self.indices(z)
        embeddings = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        codebook = torch.nn.functional.mse_loss(embeddings.float(), z.detach().float())
        commitment = torch.nn.functional.mse_loss(
            z.float(), embeddings.detach().float()
        )
        straight_through = embeddings.detach() + (z.float() - z.float().detach())
        return straight_through, indices, codebook, commitment


class Decoder128(nn.Module):
    """Phase9/13 large decoder with a parameter-free 18x8-to-10x8 fit."""

    def __init__(self, config: VQ128Config = VQ128Config()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 256, (3, 1)),  # 16x8 -> 18x8
            *[Residual(256) for _ in range(4)],
            nn.ReLU(),
            # Keep the established 10x8 decoder stem and 20/40/80 upsampling.
            # This interpolation is parameter-free; the 16 input rows have
            # overlapping learned transposed-convolution paths into this map.
            nn.Upsample(size=(10, 8), mode="bilinear", align_corners=False),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            Residual(128),
            Residual(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            Residual(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, quantized):
        if quantized.ndim != 4 or tuple(quantized.shape[1:]) != (64, 16, 8):
            raise ValueError("Expected N x 64 x 16 x 8 codebook vectors")
        output = self.layers(quantized)
        if tuple(output.shape[1:]) != (1, 80, 64):
            raise RuntimeError(
                f"Phase21 decoder produced {tuple(output.shape[1:])}, expected 1x80x64"
            )
        return output


class Face24VQVAE128(nn.Module):
    def __init__(self, config: VQ128Config = VQ128Config()):
        super().__init__()
        self.config = config
        self.encoder = Encoder128(config)
        self.quantizer = VectorQuantizer128(config)
        self.decoder = Decoder128(config)

    def forward(self, image):
        z = self.encoder(image)
        quantized, indices, codebook, commitment = self.quantizer(z)
        return self.decoder(quantized), indices, codebook, commitment


class DecoderOnly128(nn.Module):
    def __init__(self, config: VQ128Config = VQ128Config()):
        super().__init__()
        self.config = config
        self.codebook = nn.Embedding(config.entries, config.embedding_dim)
        self.decoder = Decoder128(config)
        self.codebook.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def forward(self, indices):
        if (
            indices.ndim != 3
            or tuple(indices.shape[1:]) != (16, 8)
            or indices.dtype != torch.long
            or indices.min() < 0
            or indices.max() > 255
        ):
            raise ValueError("Expected N x 16 x 8 long indices in [0,255]")
        z = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z)


def parameter_counts(model):
    return {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "codebook": sum(p.numel() for p in model.quantizer.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "full": sum(p.numel() for p in model.parameters()),
    }


def validate_indices(indices):
    if (
        indices.ndim != 3
        or tuple(indices.shape[1:]) != (16, 8)
        or indices.dtype != torch.long
        or indices.min() < 0
        or indices.max() > 255
    ):
        raise ValueError("Expected N x 16 x 8 long indices in [0,255]")


def serialize_indices(indices) -> bytes:
    """Serialize one 16x8 latent grid as 128 unsigned bytes in row-major order."""
    import numpy as np

    value = torch.as_tensor(indices)
    if value.ndim != 2 or tuple(value.shape) != (16, 8):
        raise ValueError("Expected one 16 x 8 latent index grid")
    if value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Latent indices must use an integer dtype")
    if value.numel() != 128 or int(value.min()) < 0 or int(value.max()) > 255:
        raise ValueError("Latent indices must be in [0,255]")
    payload = (
        value.to(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
        .tobytes(order="C")
    )
    if len(payload) != 128:
        raise RuntimeError("Phase21 codec did not serialize to exactly 128 bytes")
    return payload


def encode_array(values, model):
    import numpy as np

    pixels = np.asarray(values, dtype=np.float32)
    if (
        pixels.size != 5120
        or not np.isfinite(pixels).all()
        or pixels.min() < 0
        or pixels.max() > 1
    ):
        raise ValueError("Expected 5120 finite Face24 pixels in [0,1]")
    model.eval()
    x = torch.from_numpy(pixels.copy().reshape(1, 1, 80, 64)).to(
        next(model.parameters()).device
    )
    indices = model.quantizer.indices(model.encoder(x))[0].to(torch.uint8).cpu().numpy()
    payload = serialize_indices(indices)
    if len(payload) != 128:
        raise RuntimeError("Phase21 encoder did not produce exactly 128 bytes")
    return payload


def decode_indices(payload: bytes) -> torch.Tensor:
    import numpy as np

    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 128:
        raise ValueError("Phase21 payload must contain exactly 128 bytes")
    return torch.from_numpy(
        np.frombuffer(payload, dtype=np.uint8).copy().reshape(1, 16, 8).astype(np.int64)
    )


def decode_bytes(payload: bytes, decoder: DecoderOnly128):
    indices = decode_indices(payload).to(next(decoder.parameters()).device)
    decoder.eval()
    with torch.inference_mode():
        image = decoder(indices)
    return image[0, 0].detach().cpu().numpy()


def _config_from(bundle):
    if (
        bundle.get("format_version") != 1
        or bundle.get("codec") != "vqface-face24-128-v1"
        or bundle.get("kind") not in {"vqvae128", "decoder128"}
    ):
        raise ValueError("Unsupported Phase21 128-byte model artifact")
    return VQ128Config(**bundle["config"])


def load_full(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = Face24VQVAE128(_config_from(bundle))
    model.load_state_dict(bundle["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Phase21 checkpoint contains nonfinite weights")
    return model.to(device).eval(), bundle


def load_decoder(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = DecoderOnly128(_config_from(bundle))
    model.load_state_dict(bundle["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Phase21 decoder contains nonfinite weights")
    return model.to(device).eval(), bundle
