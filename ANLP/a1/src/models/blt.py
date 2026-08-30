"""Byte Latent Transformer (BLT) local encoder/decoder patch modules.

Token-free processing: instead of a linguistic vocabulary, the model consumes
raw byte values (0-255) directly.

- LocalByteEncoder: groups raw bytes into fixed patches of ``patch_size``
  bytes, embeds each byte, runs a small transformer over the ``patch_size``
  positions of each patch, and mean-pools (mask-aware) to a single latent
  vector per patch, projected to the global model dimension.
- LocalByteDecoder: takes one latent per patch and expands it back into
  ``patch_size`` byte distributions (one per byte position) using learned
  slot queries that cross-attend to the patch latent.

Padding byte id: ``BYTE_PAD = 256`` (outside the real 0-255 range).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .attention import FeedForward, MultiHeadAttention, TransformerEncoderLayer
from .norm import LayerNorm

BYTE_PAD = 256
N_BYTES = 256


class LocalByteEncoder(nn.Module):
    """Patches raw bytes (B, L) -> (B, L/patch_size, d_model) latents."""

    def __init__(
        self,
        byte_dim: int = 64,
        patch_size: int = 4,
        d_model: int = 256,
        n_local_layers: int = 2,
        n_local_heads: int = 4,
        dropout: float = 0.1,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        assert byte_dim % n_local_heads == 0
        self.byte_dim = byte_dim
        self.patch_size = patch_size
        # 256 real byte values + one learned pad embedding (id 256).
        self.byte_embed = nn.Embedding(N_BYTES + 1, byte_dim, padding_idx=BYTE_PAD)
        self.local_pos = nn.Embedding(patch_size, byte_dim)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                byte_dim,
                n_local_heads,
                n_kv_heads=None,  # MHA inside the patch
                d_ff=4 * byte_dim,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(n_local_layers)
        )
        self.proj = nn.Linear(byte_dim, d_model)

    def forward(
        self, byte_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """byte_ids: (B, L) with values in 0..255 and BYTE_PAD for padding.

        Returns (latents (B, N, d_model), patch_mask (B, N) True=has real byte).
        """
        B, L = byte_ids.shape
        P = self.patch_size
        rem = (-L) % P
        if rem:
            byte_ids = F.pad(byte_ids, (0, rem), value=BYTE_PAD)
        Lp = byte_ids.size(1)
        N = Lp // P

        x = byte_ids.view(B, N, P)
        pad_tok = x == BYTE_PAD  # (B, N, P)
        e = self.byte_embed(x) + self.local_pos.weight[None, None, :]

        # Key mask: attend to real bytes.  Fully-padded patches must keep one
        # key unmasked, otherwise their attention rows are all -inf -> NaN
        # values *and* NaN gradients through the softmax backward.
        key_ok = ~pad_tok
        empty = ~key_ok.any(dim=2, keepdim=True)  # (B, N, 1)
        slot0 = torch.arange(P, device=x.device)[None, None, :] == 0
        key_ok = key_ok | (empty & slot0)

        # (B, N, P) -> (BN, 1, 1, P), broadcast over heads and query slots.
        BN = B * N
        flat = e.reshape(BN, P, self.byte_dim)
        flat_mask = key_ok.reshape(BN, 1, 1, P)
        for layer in self.layers:
            flat, _ = layer(flat, mask=flat_mask)
        e = flat.reshape(B, N, P, self.byte_dim)

        wts = (~pad_tok).float()  # (B, N, P)
        # Fully-padded patches produce all-(-inf) attention rows (NaN outputs);
        # zero every padded slot before pooling so NaN * 0 cannot contaminate.
        e = torch.where(pad_tok.unsqueeze(-1), torch.zeros_like(e), e)
        latents = (e * wts.unsqueeze(-1)).sum(dim=2) / wts.sum(dim=2, keepdim=True).clamp(min=1.0)
        patch_mask = (~pad_tok).any(dim=2)
        # Empty patches -> zero latent (masked out by the global encoder).
        latents = torch.where(patch_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        return self.proj(latents), patch_mask


class LocalByteDecoder(nn.Module):
    """Expands one latent per patch back to patch_size byte distributions."""

    def __init__(
        self,
        byte_dim: int = 64,
        patch_size: int = 4,
        d_model: int = 256,
        n_local_heads: int = 4,
        dropout: float = 0.1,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        assert byte_dim % n_local_heads == 0
        self.byte_dim = byte_dim
        self.patch_size = patch_size
        self.latent_proj = nn.Linear(d_model, byte_dim)
        # One learned query slot per byte position inside a patch.
        self.slot_query = nn.Parameter(0.02 * torch.randn(patch_size, byte_dim))
        self.slot_pos = nn.Embedding(patch_size, byte_dim)
        self.self_attn = MultiHeadAttention(byte_dim, n_local_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(byte_dim, n_local_heads, dropout=dropout)
        self.ffn = FeedForward(byte_dim, 4 * byte_dim, dropout)
        self.norm1 = norm_layer(byte_dim)
        self.norm2 = norm_layer(byte_dim)
        self.norm3 = norm_layer(byte_dim)
        self.drop = nn.Dropout(dropout)
        self.byte_head = nn.Linear(byte_dim, N_BYTES)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory: (B, N, d_model) global hidden states (one per patch).

        Returns byte logits (B, N * patch_size, 256), ordered by byte index.
        """
        B, N, _ = memory.shape
        P = self.patch_size
        z = self.latent_proj(memory)  # (B, N, byte_dim)

        x = (self.slot_query[None, :] + self.slot_pos.weight[None, :])  # (1, P, bd)
        x = x.expand(B, N, P, -1)
        BN = B * N
        xs = x.reshape(BN, P, self.byte_dim)
        zs = z.reshape(BN, 1, self.byte_dim)

        h = self.norm1(xs)
        a, _, _ = self.self_attn(h, h, h)
        xs = xs + self.drop(a)
        h = self.norm2(xs)
        c, _, _ = self.cross_attn(h, zs, zs)
        xs = xs + self.drop(c)
        xs = xs + self.drop(self.ffn(self.norm3(xs)))

        logits = self.byte_head(xs)  # (BN, P, 256)
        return logits.reshape(B, N * P, N_BYTES)
