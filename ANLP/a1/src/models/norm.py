"""Custom Layer Normalization and RMS Normalization modules from scratch."""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Standard Layer Normalization implemented from basic PyTorch operations.

    LN(x) = ((x - mean) / sqrt(var + eps)) * weight + bias
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization implemented from scratch.

    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * scale
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


class PreLayerNorm(nn.Module):
    """Pre-layer normalization wrapper."""

    def __init__(self, dim: int, norm_type: str = "layer", eps: float = 1e-5):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(dim, eps=eps)
        else:
            self.norm = LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
