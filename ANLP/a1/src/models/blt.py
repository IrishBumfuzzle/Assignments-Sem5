"""Byte Latent Transformer (BLT) local encoder/decoder with *dynamic* patches.

Token-free processing: instead of a linguistic vocabulary, the model consumes
raw byte values (0-255) directly.  Bytes are grouped into patches of
*variable* size -- normally entropy-based dynamic patching (see
``src/entropy_patching.py``), optionally fixed stride for the ablation control
-- and the architecture follows the BLT paper (arXiv:2412.09871) in
simplified form:

- LocalByteEncoder: embeds every byte (embedding + rolling-hash byte n-gram
  hash embeddings, n = 3..8, summed and divided by the number of n-gram sizes
  + 1 as in the paper), runs ``n_local_layers`` alternating blocks of
  (a) a local-window causal transformer layer over all bytes of the line and
  (b) a cross-attention pooling block whose per-patch queries attend only to
  the bytes of their own patch (Perceiver-style, paper Section 3.2.2).
  Output: one latent vector per patch.
- LocalByteDecoder: roles reversed (paper Section 3.3) -- byte slots (the
  encoder's final byte representations plus an intra-patch slot-offset
  embedding) query the projected patch latents, attending only to the latent
  of their own patch, followed by a local-window causal transformer layer.
  Output: one byte distribution per source byte position (1:1 alignment).

Patch structure per line: ``patch_starts`` / ``patch_lens`` (right-padded
across a batch; padded entries have start = 2**31, len = 0).

Padding byte id: ``BYTE_PAD = 256`` (outside the real 0-255 range).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .attention import FeedForward, MultiHeadAttention, TransformerEncoderLayer
from .norm import LayerNorm

BYTE_PAD = 256
N_BYTES = 256

# Rolling polynomial hash constants for byte n-gram hash embeddings.
_HASH_BASE = 131
_HASH_MOD = 2_147_483_647  # 2^31 - 1 (Mersenne prime)


class CrossAttention(nn.Module):
    """Multi-head cross-attention where keys/values live in a different
    (byte) space than the queries: separate K/V projections into the query
    dimension, per the BLT paper's local-attention blocks."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        if kv_dim != q_dim:
            self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
            self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.attn = MultiHeadAttention(q_dim, n_heads, dropout=dropout)
        self.q_norm = LayerNorm(q_dim)
        self.kv_norm = LayerNorm(kv_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv_n = self.kv_norm(kv)  # pre-LayerNorm on the byte-side vectors
        if self.kv_dim != self.q_dim:
            k, v = self.k_proj(kv_n), self.v_proj(kv_n)
        else:
            k, v = kv_n, kv_n
        out, _, _ = self.attn(self.q_norm(q), k, v, mask=mask)
        return out


def rolling_ngram_indices(byte_ids: torch.Tensor, ns: Tuple[int, ...],
                          table_size: int) -> List[torch.Tensor]:
    """Per-n-gram rolling polynomial hash indices (B, L) for each n in ``ns``.

    h_n[i] = sum_{k=0}^{n-1} b[i-k] * BASE^(n-1-k)  (mod P), i.e. the hash of
    the n-gram *ending* at position i; positions with i < n-1 (incomplete
    grams) are set to 0 and must be masked out by the caller (the paper omits
    n-grams of size n or larger when i < n).
    """
    B, L = byte_ids.shape
    idxs = []
    for n in ns:
        h = torch.zeros(B, L, dtype=torch.int64, device=byte_ids.device)
        for k in range(n):
            if k == 0:
                b = byte_ids.to(torch.int64)
            else:
                b = torch.zeros_like(byte_ids, dtype=torch.int64)
                b[:, k:] = byte_ids[:, : L - k].to(torch.int64)
            h = (h * _HASH_BASE + b) % _HASH_MOD
        if n > 1:
            h[:, : n - 1] = 0
        idxs.append(h % table_size)
    return idxs


class _BandedCausalCache:
    """Cached (L, L) local-window causal mask: byte i attends to the window of
    ``window`` preceding bytes including itself (may cross patch boundaries,
    never document boundaries -- right padding keeps each row its own line)."""

    def __init__(self, window: int):
        self.window = window
        self._cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def __call__(self, L: int, device) -> torch.Tensor:
        key = (L, device)
        m = self._cache.get(key)
        if m is None:
            i = torch.arange(L)
            m = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
            m = m.to(device)
            self._cache[key] = m
        return m


class LocalByteEncoder(nn.Module):
    """Variable-size patches: raw bytes (B, L) -> (B, Np, d_model) latents.

    ``patch_starts`` / ``patch_lens``: (B, Np) per-row patch structure
    (padded entries: start=2**31, len=0).  Returns the final byte
    representations too (the decoder initialises its slots from them, as in
    the paper).
    """

    def __init__(
        self,
        byte_dim: int = 64,
        d_model: int = 256,
        n_local_layers: int = 2,
        n_local_heads: int = 4,
        max_patch: int = 12,
        ngram_ns: Tuple[int, ...] = (3, 4, 5, 6, 7, 8),
        ngram_table: int = 4096,
        window: int = 16,
        dropout: float = 0.1,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        assert byte_dim % n_local_heads == 0 and d_model % n_local_heads == 0
        self.byte_dim = byte_dim
        self.max_patch = max_patch
        self.ngram_ns = ngram_ns
        self.ngram_norm = float(len(ngram_ns) + 1)  # paper: / (num n-gram sizes + 1)
        # 256 real byte values + one learned pad embedding (id 256).
        self.byte_embed = nn.Embedding(N_BYTES + 1, byte_dim, padding_idx=BYTE_PAD)
        self.ngram_embeds = nn.ModuleList(
            [nn.Embedding(ngram_table, byte_dim) for _ in ngram_ns]
        )
        self.pool_proj = nn.Linear(byte_dim, d_model)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                byte_dim, n_local_heads, None, 4 * byte_dim, dropout, norm_layer
            )
            for _ in range(n_local_layers)
        )
        self.cross = nn.ModuleList(
            [CrossAttention(d_model, byte_dim, n_local_heads, dropout)
             for _ in range(n_local_layers)]
        )
        self.out_norm = norm_layer(d_model)
        self.band = _BandedCausalCache(window)

    # -- patch structure helpers -------------------------------------------
    @staticmethod
    def patch_of_byte(byte_ids: torch.Tensor, byte_mask: torch.Tensor,
                      patch_starts: torch.Tensor,
                      patch_lens: torch.Tensor) -> torch.Tensor:
        """pid (B, L): patch index of each byte, -1 for padding positions."""
        B, L = byte_ids.shape
        Np = patch_starts.size(1)
        pos = torch.arange(L, device=byte_ids.device).expand(B, L).contiguous()
        pid = torch.searchsorted(patch_starts, pos, right=True) - 1
        pid = pid.clamp(min=0, max=Np - 1)
        starts_g = patch_starts.gather(1, pid)
        lens_g = patch_lens.gather(1, pid)
        valid = byte_mask & (pos >= starts_g) & (pos < starts_g + lens_g)
        return torch.where(valid, pid, torch.full_like(pid, -1))

    def forward(
        self, byte_ids: torch.Tensor, patch_starts: torch.Tensor,
        patch_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (latents (B, Np, d_model), patch_mask (B, Np),
        pid (B, L), h_last (B, L, byte_dim))."""
        B, L = byte_ids.shape
        Np = patch_starts.size(1)
        device = byte_ids.device
        byte_mask = byte_ids != BYTE_PAD
        patch_mask = patch_lens > 0

        # -- embeddings: byte + rolling-hash n-gram hash embeddings ---------
        e = self.byte_embed(byte_ids)
        ngram_idx = rolling_ngram_indices(byte_ids, self.ngram_ns,
                                          self.ngram_embeds[0].num_embeddings)
        pos = torch.arange(L, device=device)
        for n, emb, idx in zip(self.ngram_ns, self.ngram_embeds, ngram_idx):
            valid = pos >= (n - 1)  # (L,) - incomplete grams contribute 0
            e = e + emb(idx) * valid[None, :, None].to(e.dtype)
        e = e / self.ngram_norm

        # -- masks -----------------------------------------------------------
        # Every query row must keep at least one allowed key, or the softmax
        # over all -inf produces NaN whose gradients poison the whole batch.
        pid = self.patch_of_byte(byte_ids, byte_mask, patch_starts, patch_lens)
        diag = torch.eye(L, dtype=torch.bool, device=device)
        # local (byte self-)attention: banded causal window; keys = real bytes
        # (the diagonal is always allowed so padded queries stay finite).
        local_mask = self.band(L, device).unsqueeze(0).unsqueeze(1) & \
            (byte_mask.view(B, 1, 1, L) | diag.unsqueeze(0).unsqueeze(0))
        # cross-attention pooling: patch j attends only to bytes of patch j;
        # padded patches get a dummy key (byte 0) so their rows stay finite.
        key0 = torch.zeros(L, dtype=torch.bool, device=device); key0[0] = True
        enc_x_mask = (pid.unsqueeze(1) == torch.arange(Np, device=device).view(1, Np, 1))
        enc_x_mask = enc_x_mask & byte_mask.unsqueeze(1)  # (B, Np, L)
        enc_x_mask = enc_x_mask | (~patch_mask.unsqueeze(-1) & key0.view(1, 1, L))

        # -- init patch queries by pooling the (masked) byte embeddings ------
        w = enc_x_mask.to(e.dtype)
        sum_x = torch.bmm(w, e)
        cnt = w.sum(dim=2, keepdim=True).clamp(min=1.0)
        P = self.pool_proj(sum_x / cnt)
        P = torch.where(patch_mask.unsqueeze(-1), P, torch.zeros_like(P))

        h = e
        for l in range(len(self.layers)):
            h, _ = self.layers[l](h, mask=local_mask)
            h = torch.where(byte_mask.unsqueeze(-1), h, torch.zeros_like(h))
            c = self.cross[l](P, h, mask=enc_x_mask.unsqueeze(1))
            P = torch.where(patch_mask.unsqueeze(-1), P + c, torch.zeros_like(P))
        latents = self.out_norm(P)
        return latents, patch_mask, pid, h


class LocalByteDecoder(nn.Module):
    """Expands one latent per patch back to per-byte distributions.

    Byte slots are queries, the (projected) patch latents are keys/values;
    each slot attends only to the latent of the patch it belongs to, then a
    local-window causal transformer layer mixes neighbouring slots.  Intra-patch
    slot-offset embeddings disambiguate positions within a patch.
    """

    def __init__(
        self,
        byte_dim: int = 64,
        d_model: int = 256,
        max_patch: int = 12,
        n_local_heads: int = 4,
        n_local_layers: int = 1,
        window: int = 16,
        dropout: float = 0.1,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or LayerNorm
        assert byte_dim % n_local_heads == 0
        self.byte_dim = byte_dim
        self.max_patch = max_patch
        self.latent_proj = nn.Linear(d_model, byte_dim, bias=False)
        # Slot offset embedding: position of the byte inside its patch.
        self.slot_pos = nn.Embedding(max_patch, byte_dim)
        self.cross = nn.ModuleList(
            [MultiHeadAttention(byte_dim, n_local_heads, dropout=dropout)
             for _ in range(n_local_layers)]
        )
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                byte_dim, n_local_heads, None, 4 * byte_dim, dropout, norm_layer
            )
            for _ in range(n_local_layers)
        )
        self.q_norm = norm_layer(byte_dim)
        self.kv_norm = norm_layer(byte_dim)
        self.out_norm = norm_layer(byte_dim)
        self.drop = nn.Dropout(dropout)
        self.byte_head = nn.Linear(byte_dim, N_BYTES)
        self.band = _BandedCausalCache(window)

    def forward(
        self, memory: torch.Tensor, h_enc: torch.Tensor, byte_mask: torch.Tensor,
        pid: torch.Tensor, patch_starts: torch.Tensor,
    ) -> torch.Tensor:
        """memory: (B, Np, d_model) global hidden states; h_enc: (B, L, byte_dim)
        final encoder byte representations; pid: (B, L) byte -> patch map.
        Returns byte logits (B, L, 256) aligned 1:1 to the source bytes."""
        B, L, _ = h_enc.shape      # L = byte length (memory is patch-level)
        device = memory.device
        Np = memory.size(1)

        # byte -> its patch's latent (single-key cross-attention):
        # dec_x_mask[b, i, j] = True iff real byte i belongs to patch j
        latent_kv = self.latent_proj(memory)  # (B, Np, byte_dim)
        real = pid >= 0
        dec_x_mask = torch.zeros(B, L, Np, dtype=torch.bool, device=device)
        rb = torch.arange(B, device=device)[:, None].expand(B, L).reshape(-1)
        lb = torch.arange(L, device=device)[None, :].expand(B, L).reshape(-1)
        pp = pid.reshape(-1).clamp(min=0)
        dec_x_mask[rb[real.reshape(-1)], lb[real.reshape(-1)],
                   pp[real.reshape(-1)]] = True
        # padded byte slots get a dummy key (patch 0) so rows stay finite
        dummy_key = torch.zeros(Np, dtype=torch.bool, device=device); dummy_key[0] = True
        dec_x_mask = dec_x_mask | (~real.unsqueeze(-1) & dummy_key.view(1, 1, Np))

        # intra-patch slot offset (0 for padding slots; masked out below)
        pos = torch.arange(L, device=device)
        starts_g = patch_starts.gather(1, pid.clamp(min=0))
        offset = (pos - starts_g).clamp(min=0, max=self.max_patch - 1)
        offset = torch.where(real, offset, torch.zeros_like(offset))

        d = h_enc + self.slot_pos(offset)  # (B, L, byte_dim)
        d = torch.where(byte_mask.unsqueeze(-1), d, torch.zeros_like(d))
        diag = torch.eye(L, dtype=torch.bool, device=device)
        local_mask = self.band(L, device).unsqueeze(0).unsqueeze(1) & \
            (byte_mask.view(B, 1, 1, L) | diag.unsqueeze(0).unsqueeze(0))

        for l in range(len(self.layers)):
            q = self.q_norm(d)
            kv = self.kv_norm(latent_kv)
            c, _, _ = self.cross[l](q, kv, kv, mask=dec_x_mask.unsqueeze(1))
            d = torch.where(byte_mask.unsqueeze(-1),
                            d + self.drop(c), torch.zeros_like(d))
            d, _ = self.layers[l](d, mask=local_mask)
            d = torch.where(byte_mask.unsqueeze(-1), d, torch.zeros_like(d))
        return self.byte_head(self.out_norm(d))  # (B, L, 256)
