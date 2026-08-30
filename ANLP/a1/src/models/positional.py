"""Positional encodings implemented from scratch.

- SinusoidalPositionalEncoding: additive absolute sin/cos positional encoding
  (Vaswani et al., 2017).
- RotaryPositionalEmbedding (RoPE): rotates query/key vectors by
  position-dependent angles (Su et al., 2021).  Encodes *relative* position
  because the rotation difference between positions i and j depends only on
  (i - j).
"""

import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Additive sinusoidal absolute positional encoding.

    ``pe(pos, 2i)   = sin(pos / 10000^(2i/d))``
    ``pe(pos, 2i+1) = cos(pos / 10000^(2i/d))``
    """

    def __init__(self, dim: int, max_len: int = 4096, min_freq: float = 1.0,
                 max_freq: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal encoding"
        half = dim // 2
        inv_freq = 1.0 / (
            (max_freq / min_freq) ** (torch.arange(0, half, dtype=torch.float32) / half)
        )
        pos = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(pos, inv_freq)  # (max_len, dim/2)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, L, D).  Adds pe[offset : offset+L] to each position."""
        L = x.size(1)
        assert offset + L <= self.max_len, (
            f"sequence length {offset + L} exceeds max_len {self.max_len}"
        )
        return x + self.pe[offset : offset + L].to(x.dtype).unsqueeze(0)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (NeoX-style half-rotation) for attention queries/keys.

    The head vector is split in half: (x1, x2).  For position ``p`` the
    rotated vector is ``(x1*cos(p*theta) - x2*sin(p*theta),
    x2*cos(p*theta) + x1*sin(p*theta))`` where
    ``theta_i = 10000^(-2i/d)``.
    """

    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_len = max_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _cos_sin(self, positions: torch.Tensor):
        # positions: (L,)  ->  cos, sin: (L, head_dim/2)
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        return freqs.cos(), freqs.sin()

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate a set of head vectors.

        x: (B, H, L, head_dim), positions: (L,) absolute positions.
        """
        assert x.size(2) <= self.max_len
        cos, sin = self._cos_sin(positions)  # (L, d/2)
        cos = cos[None, None, :, :]  # (1, 1, L, d/2)
        sin = sin[None, None, :, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1).to(x.dtype)

    def positions(self, start: int, length: int, device) -> torch.Tensor:
        return torch.arange(start, start + length, device=device)
