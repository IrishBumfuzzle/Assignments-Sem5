"""Custom Attention modules: Scaled Dot-Product Attention, MHA, GQA, and FFN."""

import math
from typing import Optional
import torch
from torch import nn

from .positional import RoPE


class SDPA(nn.Module):
    """Scaled Dot-Product Attention from scratch.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
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
    ):
        """Args:

        q: [B, H, T_q, D]
        k: [B, H, T_k, D]
        v: [B, H, T_k, D]
        mask: boolean mask where True = keep/attend, False = mask out,
              or additive float mask where 0 = keep, -inf = mask out.
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Handle mask shapes properly for [B, H, T_q, T_k]
            if mask.dim() == 2:
                b, t_q, t_k = scores.size(0), scores.size(-2), scores.size(-1)
                if mask.size(0) == t_q and mask.size(1) == t_k and mask.size(0) != b:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T_q, T_k]
                elif mask.size(0) == b and mask.size(1) == t_k and mask.size(0) != t_q:
                    mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_k]
                elif mask.size(0) == b and mask.size(1) == t_k and b == t_q:
                    # B == T_q == T_k: check if it's a lower-triangular causal mask
                    if torch.equal(mask, torch.tril(torch.ones_like(mask))):
                        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T_q, T_k]
                    else:
                        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_k]
                elif mask.size(0) == t_q and mask.size(1) == t_k:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                else:
                    mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, T_q, T_k]

            if mask.dtype == torch.bool:
                neg_inf = -1e4 if scores.dtype == torch.float16 else -1e9
                scores = scores.masked_fill(~mask, neg_inf)
            else:
                scores = scores + mask

        weights = torch.softmax(scores, dim=-1)
        # Apply dropout to attention weights if training
        if self.dropout.p > 0 and self.training:
            weights_dropped = self.dropout(weights)
            output = torch.matmul(weights_dropped, v)
        else:
            output = torch.matmul(weights, v)
        return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (MHA) module from scratch."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attention = SDPA(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, head_dim]
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        offset: int = 0,
        return_weights: bool = False,
    ):
        is_self_attn = context is None
        context = x if is_self_attn else context

        b, tq, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))

        if self.rope is not None:
            q, k = self.rope(q, k, offset=offset)

        out, weights = self.attention(q, k, v, mask=mask)
        # [B, H, T_q, head_dim] -> [B, T_q, D]
        out = out.transpose(1, 2).contiguous().view(b, tq, self.dim)
        out = self.out_proj(out)

        return (out, weights) if return_weights else out


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) module from scratch."""

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        rope: Optional[RoPE] = None,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if heads % kv_heads != 0:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_heads})")

        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.num_queries_per_kv = heads // kv_heads
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attention = SDPA(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        offset: int = 0,
        return_weights: bool = False,
    ):
        is_self_attn = context is None
        context = x if is_self_attn else context

        b, tq, _ = x.shape
        tk = context.size(1)

        q = self.q_proj(x).view(b, tq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, tk, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, tk, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, offset=offset)

        # Expand KV heads to match query heads: [B, H_kv, T_k, D_h] -> [B, H_q, T_k, D_h]
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        out, weights = self.attention(q, k, v, mask=mask)
        # [B, H, T_q, head_dim] -> [B, T_q, D]
        out = out.transpose(1, 2).contiguous().view(b, tq, self.dim)
        out = self.out_proj(out)

        return (out, weights) if return_weights else out


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
