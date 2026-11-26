"""64-byte codec (Phase 18): 8x8 VQ encoder/codebook plus the large decoder."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .layers import Residual, VQConfig, VQEncoder, VectorQuantizer


@dataclass(frozen=True)
class LargeConfig:
    architecture: str = "vqface-decoder-large-v1"
    encoder_architecture: str = "vqface-v1"
    channels: tuple = (256, 128, 64)
    residual_blocks: tuple = (4, 2, 1)
    embedding_dim: int = 64
    entries: int = 256
    grid: tuple = (8, 8)
    image_size: tuple = (64, 80)
    order: str = "row-major-yx"

    def __post_init__(self):
        expected = {"architecture": "vqface-decoder-large-v1", "encoder_architecture": "vqface-v1",
                    "channels": (256, 128, 64), "residual_blocks": (4, 2, 1), "embedding_dim": 64, "entries": 256,
                    "grid": (8, 8), "image_size": (64, 80), "order": "row-major-yx"}
        for key, value in expected.items():
            actual = getattr(self, key)
            if isinstance(value, tuple):
                actual = tuple(actual)
            if actual != value:
                raise ValueError("Unsupported larger-decoder configuration")


class LargeDecoder(nn.Module):
    def __init__(self, config=LargeConfig()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 256, (3, 1)),
            *[Residual(256) for _ in range(4)], nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), Residual(128), Residual(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), Residual(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 8, 8):
            raise ValueError("Expected N x 64 x 8 x 8 embeddings")
        return self.layers(z)


def validate_indices(indices):
    if (indices.ndim != 3 or tuple(indices.shape[1:]) != (8, 8) or indices.dtype != torch.long
            or indices.min() < 0 or indices.max() > 255):
        raise ValueError("Expected N x 8 x 8 long indices in [0,255]")


class LargeDecoderOnly(nn.Module):
    def __init__(self, config=LargeConfig()):
        super().__init__()
        self.config = config
        self.codebook = nn.Embedding(256, 64)
        self.codebook.requires_grad_(False)
        self.decoder = LargeDecoder(config)

    def forward(self, indices):
        validate_indices(indices)
        return self.decoder(self.codebook(indices).permute(0, 3, 1, 2).contiguous())


class Face24LargeVQVAE(nn.Module):
    def __init__(self, config=VQConfig(), decoder_config=LargeConfig()):
        super().__init__()
        self.config = config
        self.decoder_config = decoder_config
        self.encoder = VQEncoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = LargeDecoder(decoder_config)

    def forward(self, image):
        z = self.encoder(image)
        quantized, indices, codebook, commitment = self.quantizer(z)
        return self.decoder(quantized), indices, codebook, commitment


def parameter_counts(model):
    return {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "codebook": sum(p.numel() for p in model.quantizer.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
        "full": sum(p.numel() for p in model.parameters()),
    }


def load_full(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("format_version") != 1 or bundle.get("codec") != "vqface-face24-large-v1" or bundle.get("kind") != "vqvae":
        raise ValueError("Unsupported Phase13 full checkpoint")
    config = VQConfig(**bundle["vq_config"])
    decoder_config = LargeConfig(**bundle["decoder_config"])
    model = Face24LargeVQVAE(config, decoder_config)
    model.load_state_dict(bundle["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite Phase13 checkpoint")
    return model.to(device).eval(), bundle


def load_decoder(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1 or bundle.get("codec") != "vqface-face24-large-v1" or bundle.get("kind") != "decoder":
        raise ValueError("Unsupported Phase13 decoder artifact")
    vq_config = VQConfig(**bundle["vq_config"])
    decoder_config = LargeConfig(**bundle["decoder_config"])
    model = LargeDecoderOnly(decoder_config)
    model.load_state_dict(bundle["state_dict"], strict=True)
    if model.codebook.weight.shape != (vq_config.entries, vq_config.embedding_dim):
        raise ValueError("Decoder codebook disagrees with VQ configuration")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval(), bundle


@torch.inference_mode()
def encode_array(values, model):
    """64 unsigned codebook indices: row-major grid[y,x], no per-image metadata."""
    values = np.asarray(values, dtype=np.float32)
    if values.size != 5120 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected 5120 finite pixels in [0,1]")
    model.eval()
    x = torch.from_numpy(values.copy().reshape(1, 1, 80, 64)).to(next(model.parameters()).device)
    indices = model.quantizer.indices(model.encoder(x))
    return indices[0].to(torch.uint8).cpu().numpy().tobytes(order="C")


def decode_bytes(payload, decoder, device=None):
    if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) != 64:
        raise ValueError("Phase13 payload must contain exactly 64 bytes")
    validate_indices(torch.zeros((1, 8, 8), dtype=torch.long))
    target_device = device or next(decoder.parameters()).device
    indices = torch.tensor(list(payload), dtype=torch.long, device=target_device).reshape(1, 8, 8)
    with torch.inference_mode():
        out = decoder(indices)
    if tuple(out.shape) != (1, 1, 80, 64):
        raise ValueError("Phase13 decoder output must be N x 1 x 80 x 64")
    return out[0, 0].detach().cpu().numpy()
