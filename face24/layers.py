"""Shared VQ building blocks: 8x8 grid, 256 entries, 64-dimensional embeddings."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VQConfig:
    architecture: str = "vqface-v1"
    embedding_dim: int = 64
    entries: int = 256
    grid: tuple = (8, 8)
    image_size: tuple = (64, 80)
    order: str = "row-major-yx"

    def __post_init__(self):
        if (self.architecture != "vqface-v1" or self.embedding_dim != 64 or self.entries != 256
                or tuple(self.grid) != (8, 8) or tuple(self.image_size) != (64, 80) or self.order != "row-major-yx"):
            raise ValueError("Incompatible VQ codec configuration")


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.ReLU(), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.layers(x)


class VQEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),  # 40 x 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 20 x 16
            nn.Conv2d(64, 128, 4, 2, 1),  # 10 x 8
            Residual(128), Residual(128), nn.ReLU(),
            # Valid learned projection uses ALL ten feature rows, without
            # cropping/padding/resizing the original rectangular image.
            nn.Conv2d(128, config.embedding_dim, (3, 1)))  # 8 x 8

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 80, 64):
            raise ValueError("Expected N x 1 x 80 x 64 images")
        return self.layers(x)


class VectorQuantizer(nn.Module):
    """Standard codebook + commitment losses, not EMA; ties select first index."""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.entries, config.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / config.entries, 1 / config.entries)

    @torch.no_grad()
    def indices(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (64, 8, 8):
            raise ValueError("Expected N x 64 x 8 x 8 features")
        with torch.autocast(device_type=z.device.type, enabled=False):
            vectors = z.float().permute(0, 2, 3, 1).reshape(-1, 64)
            weights = self.embedding.weight.float()
            distance = vectors.square().sum(1, keepdim=True) + weights.square().sum(1)[None] - 2 * vectors @ weights.T
            return distance.argmin(1).reshape(-1, 8, 8)

    def forward(self, z):
        indices = self.indices(z)
        embedding = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(embedding.float(), z.detach().float())
        commitment_loss = F.mse_loss(z.float(), embedding.detach().float())
        # Exact embedding values in forward; identity gradient to encoder.
        quantized = embedding.detach() + (z.float() - z.float().detach())
        return quantized, indices, codebook_loss, commitment_loss
