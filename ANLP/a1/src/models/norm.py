"""Normalization modules implemented from scratch.

- LayerNorm: classic zero-mean / unit-variance normalization (with optional
  elementwise affine parameters).  Computed with explicit mean/variance
  operations instead of wrapping ``nn.LayerNorm``.
- RMSNorm: normalizes by root-mean-square only (no centering, no bias),
  as popularized by the LLaMA line of models.
"""

from typing import Optional

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight: Optional[nn.Parameter] = (
            nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        )
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(dim)) if elementwise_affine else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class RMSNorm(nn.Module):
    """Root-mean-square normalization: x / rms(x) * gamma (no centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def make_norm(norm: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Factory: 'layernorm' -> LayerNorm, 'rmsnorm' -> RMSNorm."""
    if norm == "layernorm":
        return LayerNorm(dim, eps=eps)
    if norm == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"Unknown normalization: {norm!r}")
