"""Sinusoidal and Rotary Positional Encodings (RoPE) from scratch."""

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        self.dim = dim
        position = torch.arange(max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )  # [dim/2]

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, dim]

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding to input tensor x [B, T, D]."""
        seq_len = x.size(1)
        return x + self.pe[:, offset : offset + seq_len, :].to(device=x.device, dtype=x.dtype)


class RoPE(nn.Module):
    """Rotary Positional Embedding (Su et al., 2021).

    Applies rotary transformation to query and key states on each attention head.
    """

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension ({dim}) must be even.")
        self.dim = dim
        self.max_len = max_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]
        t = torch.arange(max_len).float()  # [max_len]
        freqs = torch.outer(t, inv_freq)  # [max_len, dim/2]

        # Shape: [1, 1, max_len, dim/2] for broadcasting over [B, H, T, dim/2]
        self.register_buffer("cos", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _rotate(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate tensor x of shape [B, H, T, D]."""
        seq_len = x.size(-2)
        cos = self.cos[:, :, offset : offset + seq_len, :].to(device=x.device, dtype=x.dtype)
        sin = self.sin[:, :, offset : offset + seq_len, :].to(device=x.device, dtype=x.dtype)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply RoPE to both query [B, H, T_q, D] and key [B, H, T_k, D]."""
        return self._rotate(q, offset=offset), self._rotate(k, offset=offset)
