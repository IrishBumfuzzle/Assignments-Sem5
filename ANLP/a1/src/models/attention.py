"""Attention building blocks implemented from scratch.

- ScaledDotProductAttention: raw ``softmax(Q K^T / sqrt(d)) V`` with masking.
- MultiHeadAttention: MHA (n_kv_heads == n_heads) and Grouped-Query Attention
  (GQA, n_kv_heads < n_heads), with optional RoPE and a KV cache for
  incremental (autoregressive) decoding.
- FeedForward: position-wise feed-forward network (FFN).
- TransformerEncoderLayer / TransformerDecoderLayer: pre-LayerNorm /
  pre-RMSNorm residual blocks built only on the modules in this file.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .norm import LayerNorm
from .positional import RotaryPositionalEmbedding


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention over pre-split head tensors.

    All shapes are (B, H, L, d_k).  ``mask`` is a boolean tensor broadcastable
    to (B, H, L_q, L_k) where ``True`` means "allowed to attend".
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (d_k ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # Softmax in fp32 for numerical stability, then cast back.
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention supporting MHA and GQA.

    - MHA: ``n_kv_heads=None`` (or ``n_kv_heads == n_heads``).
    - GQA: ``n_kv_heads < n_heads`` (must divide ``n_heads``).  Each group of
      query heads shares one key/value head; K/V are expanded by
      ``repeat_interleave`` before attention.

    Optional RoPE: if ``rope`` is given, ``pos_q`` / ``pos_k`` (1-D absolute
    positions) must be supplied and the query/key head vectors are rotated.

    Optional KV cache (for incremental decoding): on the first call pass
    ``kv_cache=None`` and keep the returned cache; on later single-token
    calls pass the cache back in.  Cached K/V are stored *after* GQA
    expansion and RoPE, so they are ready to concatenate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        assert n_heads >= 1 and d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
        )
        self.head_dim = d_model // self.n_heads
        self.group = self.n_heads // self.n_kv_heads  # 1 for plain MHA
        self.rope = rope

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention(dropout)

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor):
        """Expand GQA kv heads to full head count: (B, n_kv, L, d) -> (B, H, L, d)."""
        if self.group == 1:
            return k, v
        return (
            k.repeat_interleave(self.group, dim=1),
            v.repeat_interleave(self.group, dim=1),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_q: Optional[torch.Tensor] = None,
        pos_k: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """query: (B, Lq, D), key/value: (B, Lk, D).

        Returns (output (B, Lq, D), attn_weights (B, H, Lq, Lk), new_cache).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H, hd, Hkv = self.n_heads, self.head_dim, self.n_kv_heads

        q = self.W_q(query).view(B, Lq, H, hd).transpose(1, 2)
        k = self.W_k(key).view(B, Lk, Hkv, hd).transpose(1, 2)
        v = self.W_v(value).view(B, Lk, Hkv, hd).transpose(1, 2)

        if self.rope is not None:
            assert pos_q is not None and pos_k is not None, (
                "RoPE requires pos_q and pos_k"
            )
            q = self.rope.apply(q, pos_q)
            k = self.rope.apply(k, pos_k)

        k_exp, v_exp = self._expand_kv(k, v)
        if kv_cache is None:
            k_out, v_out = k_exp, v_exp
        else:
            ck, cv = kv_cache
            k_out = torch.cat([ck, k_exp], dim=2)
            v_out = torch.cat([cv, v_exp], dim=2)
        cache = (k_out, v_out)

        out, attn_w = self.attn(q, k_out, v_out, mask)
        out = out.transpose(1, 2).reshape(B, Lq, -1)
        return self.W_o(out), attn_w, cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm encoder block: x + Attn(Norm(x)); x + FFN(Norm(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        norm_layer=None,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads, dropout, rope=rope
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """x: (B, L, D).  mask: broadcastable (.., Lk) True=attend.  pos: (L,) for RoPE."""
        h = self.norm1(x)
        a, attn_w, _ = self.self_attn(h, h, h, mask=mask, pos_q=pos, pos_k=pos)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, attn_w


class TransformerDecoderLayer(nn.Module):
    """Pre-norm decoder block: causal self-attention + cross-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        norm_layer=None,
        rope: Optional[RotaryPositionalEmbedding] = None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        # Self-attention (RoPE-eligible, MHA/GQA per n_kv_heads, KV-cacheable).
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, n_kv_heads, dropout, rope=rope
        )
        # Cross-attention always attends to the full encoder memory; no RoPE,
        # plain MHA (kv heads == q heads).
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.norm3 = norm_layer(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """x: (B, L, D), memory: (B, S, D).

        self_mask: (Lq, Lk) or (B, 1, Lq, Lk) True=attend (causal during
        teacher forcing; None for single-token cached steps).
        memory_mask: (B, 1, 1, S) True=attend.
        Returns (x, new_self_attn_cache).
        """
        h = self.norm1(x)
        a, _, cache = self.self_attn(
            h, h, h, mask=self_mask, pos_q=pos, pos_k=pos, kv_cache=kv_cache
        )
        x = x + self.drop(a)
        h = self.norm2(x)
        c, _, _ = self.cross_attn(h, memory, memory, mask=memory_mask)
        x = x + self.drop(c)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, cache
