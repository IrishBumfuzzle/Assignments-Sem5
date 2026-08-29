"""Byte Latent Transformer (BLT) modules: Local Encoder and Local Decoder."""

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class LocalEncoder(nn.Module):
    """Local Encoder for BLT: converts raw byte sequences into patch representations.

    Input: [B, T_bytes] -> Output: [B, T_patches, model_dim]
    """

    def __init__(
        self,
        byte_dim: int = 64,
        model_dim: int = 256,
        patch_size: int = 4,
        vocab_size: int = 260,  # 256 byte values + PAD(0), BOS(256), EOS(257), UNK(258)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.byte_dim = byte_dim
        self.model_dim = model_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, byte_dim)
        self.proj = nn.Sequential(
            nn.Linear(byte_dim * patch_size, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Args:

        byte_ids: [B, T_bytes]

        Returns:
            patch_embeddings: [B, T_patches, model_dim]
        """
        b, t = byte_ids.shape
        pad_len = (-t) % self.patch_size
        if pad_len > 0:
            byte_ids = F.pad(byte_ids, (0, pad_len), value=256)

        # [B, T_padded, byte_dim]
        embeds = self.embedding(byte_ids)
        # [B, T_patches, patch_size * byte_dim]
        patches = embeds.view(b, -1, self.patch_size * self.byte_dim)
        # [B, T_patches, model_dim]
        out = self.layer_norm(self.proj(patches))
        return out


class LocalDecoder(nn.Module):
    """Local Decoder for BLT: converts patch representations back to byte logits.

    Input: [B, T_patches, model_dim] -> Output: [B, T_bytes, vocab_size]
    """

    def __init__(
        self,
        model_dim: int = 256,
        byte_dim: int = 64,
        patch_size: int = 4,
        vocab_size: int = 260,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.byte_dim = byte_dim
        self.vocab_size = vocab_size

        self.proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, patch_size * byte_dim),
        )
        self.out = nn.Linear(byte_dim, vocab_size)

    def forward(self, patch_states: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """Args:

        patch_states: [B, T_patches, model_dim]
        target_len: optional integer length to slice the unrolled bytes.

        Returns:
            byte_logits: [B, T_bytes, vocab_size]
        """
        b, num_patches, _ = patch_states.shape
        # [B, num_patches, patch_size, byte_dim]
        x = self.proj(patch_states).view(b, num_patches * self.patch_size, self.byte_dim)
        logits = self.out(x)  # [B, num_patches * patch_size, vocab_size]

        if target_len is not None:
            logits = logits[:, :target_len, :]

        return logits
