"""Full models: encoder/decoder stacks, the Seq2Seq Transformer (C1-C4) and
the Byte Latent Transformer (C5).

All submodules are the from-scratch building blocks in this package
(attention.py, positional.py, norm.py, blt.py).
"""

import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .attention import TransformerDecoderLayer, TransformerEncoderLayer
from .blt import BYTE_PAD, LocalByteDecoder, LocalByteEncoder
from .norm import LayerNorm
from .positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding


@dataclass
class TransformerConfig:
    """Architecture hyperparameters shared by all five ablation configs."""

    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: Optional[int] = None    # None -> n_heads (MHA); < n_heads for GQA
    n_layers: int = 4
    d_ff: int = 0                # 0 -> 4 * d_model
    dropout: float = 0.1
    max_len: int = 2048
    pos_encoding: str = "sinusoidal"   # "sinusoidal" | "rope" | "none"
    norm: str = "layernorm"            # "layernorm" | "rmsnorm"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
        )

    def norm_layer(self):
        from .norm import RMSNorm
        return RMSNorm if self.norm == "rmsnorm" else LayerNorm

    def make_rope(self) -> Optional[RotaryPositionalEmbedding]:
        if self.pos_encoding == "rope":
            return RotaryPositionalEmbedding(self.d_model // self.n_heads, self.max_len)
        return None

    def describe(self) -> dict:
        return {
            "pos_encoding": self.pos_encoding,
            "attention": "gqa" if self.n_kv_heads < self.n_heads else "mha",
            "n_kv_heads": self.n_kv_heads,
            "normalization": self.norm,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }


class TransformerEncoder(nn.Module):
    """Encoder stack.

    Either embeds token ids (seq2seq) or consumes external representations
    directly (BLT latents).  Pre-norm blocks with a final norm.
    """

    def __init__(self, cfg: TransformerConfig, vocab_size: Optional[int] = None,
                 pad_idx: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embedding = (
            nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_idx)
            if vocab_size is not None
            else None
        )
        self.scale = math.sqrt(cfg.d_model)
        self.pe = (
            SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
            if cfg.pos_encoding == "sinusoidal"
            else None
        )
        self.rope = cfg.make_rope()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff or None,
                cfg.dropout, cfg.norm_layer(), rope=self.rope,
            )
            for _ in range(cfg.n_layers)
        )
        self.final_norm = cfg.norm_layer()(cfg.d_model)

    def forward(self, x, mask: Optional[torch.Tensor] = None,
                repr_input: bool = False) -> torch.Tensor:
        """x: (B, L) token ids or (B, L, D) representations.

        mask: (B, L) bool, True = real token (used to mask out padding keys).
        """
        if repr_input:
            h = x
        else:
            h = self.embedding(x) * self.scale
        if self.pe is not None:
            h = self.pe(h)
        L = h.size(1)
        attn_mask = mask.view(-1, 1, 1, L) if mask is not None else None
        pos = torch.arange(L, device=x.device)
        for layer in self.layers:
            h, _ = layer(h, mask=attn_mask, pos=pos)
        return self.final_norm(h)


class TransformerDecoder(nn.Module):
    """Decoder stack with causal self-attention, cross-attention to memory,
    and KV-cache support for incremental greedy decoding."""

    def __init__(self, cfg: TransformerConfig, vocab_size: int, pad_idx: int):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(cfg.d_model)
        self.pe = (
            SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
            if cfg.pos_encoding == "sinusoidal"
            else None
        )
        self.rope = cfg.make_rope()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff or None,
                cfg.dropout, cfg.norm_layer(), rope=self.rope,
            )
            for _ in range(cfg.n_layers)
        )
        self.final_norm = cfg.norm_layer()(cfg.d_model)

    @staticmethod
    def causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
        """(Lq, Lk) bool: query i may attend to key j iff j <= Lk - Lq + i."""
        idx_q = torch.arange(Lq, device=device)
        idx_k = torch.arange(Lk, device=device)
        return idx_k[None, :] <= (Lk - Lq + idx_q)[:, None]

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[torch.Tensor]] = None,
        pos_start: int = 0,
    ):
        """x: (B, L) token ids.  Returns (h (B, L, D), caches)."""
        h = self.embedding(x) * self.scale
        L = h.size(1)
        if self.pe is not None:
            h = self.pe(h, offset=pos_start)

        if cache is None:
            self_mask = self.causal_mask(L, L, x.device) if L > 1 else None
        else:
            # Incremental step: the single new query attends to the full
            # (already causally valid) cache; no mask needed.
            self_mask = None
        mem_mask = (
            memory_mask.unsqueeze(1).unsqueeze(2)  # (B, S) -> (B, 1, 1, S)
            if memory_mask is not None
            else None
        )
        pos = torch.arange(pos_start, pos_start + L, device=x.device)

        caches = []
        for i, layer in enumerate(self.layers):
            h, layer_cache = layer(
                h, memory,
                self_mask=self_mask, memory_mask=mem_mask,
                pos=pos,
                kv_cache=cache[i] if cache is not None else None,
            )
            caches.append(layer_cache)
        return self.final_norm(h), caches


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder Transformer (configs C1-C4)."""

    def __init__(self, src_vocab: int, tgt_vocab: int, cfg: TransformerConfig,
                 pad_idx: int, eos_idx: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(cfg, vocab_size=src_vocab, pad_idx=pad_idx)
        self.decoder = TransformerDecoder(cfg, vocab_size=tgt_vocab, pad_idx=pad_idx)
        self.out_proj = nn.Linear(cfg.d_model, tgt_vocab)
        self.eos_idx = eos_idx

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward.  src: (B, S), tgt: (B, T) (incl. BOS/EOS).

        Returns logits (B, T-1, V) aligned with tgt[:, 1:].
        """
        memory = self.encoder(src, mask=src_mask)
        h, _ = self.decoder(tgt[:, :-1], memory, memory_mask=src_mask)
        return self.out_proj(h)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int,
                 bos_idx: int) -> torch.Tensor:
        """Greedy autoregressive decoding with a per-layer KV cache.

        Returns token ids (B, T) (BOS removed, rows truncated at first EOS).
        """
        B = src.size(0)
        device = src.device
        self.eval()
        memory = self.encoder(src, mask=src_mask)
        mem_mask = src_mask  # (B, S); decoder reshapes to (B,1,1,S)

        cur = torch.full((B, 1), bos_idx, device=device, dtype=torch.long)
        cache: Optional[List[torch.Tensor]] = None
        rows: List[torch.Tensor] = []
        for t in range(max_len):
            h, cache = self.decoder(cur, memory, memory_mask=mem_mask,
                                    cache=cache, pos_start=t)
            logits = self.out_proj(h)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            rows.append(next_id)
            if (next_id == self.eos_idx).all():
                break
            # Sequences that already finished get EOS fed back (outputs are
            # truncated at the first EOS after the loop).
            done = next_id == self.eos_idx
            cur = torch.where(done, torch.full_like(next_id, self.eos_idx), next_id)
        ids = torch.cat(rows, dim=1)  # (B, T)
        # Truncate every row right after its first EOS.
        eos_pos = (ids == self.eos_idx).nonzero(as_tuple=False)
        out = []
        for i in range(B):
            e = eos_pos[eos_pos[:, 0] == i]
            last = int(e[:, 1].max().item()) if len(e) else ids.size(1)
            out.append(ids[i, : last + 1])
        return torch.nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=self.eos_idx)


class BLTModel(nn.Module):
    """Byte Latent Transformer (config C5): token-free raw-byte processing.

    raw cipher bytes -> LocalByteEncoder (patches) -> global transformer ->
    LocalByteDecoder -> byte distributions.  The plain length equals the
    cipher length byte-for-byte, so decoding is a single forward pass with no
    autoregression.
    """

    def __init__(self, cfg: TransformerConfig, patch_size: int = 4,
                 byte_dim: int = 64, n_local_layers: int = 2,
                 n_local_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        self.patch_size = patch_size
        self.local_encoder = LocalByteEncoder(
            byte_dim=byte_dim, patch_size=patch_size, d_model=cfg.d_model,
            n_local_layers=n_local_layers, n_local_heads=n_local_heads,
            dropout=cfg.dropout, norm_layer=cfg.norm_layer(),
        )
        # Global transformer over patch latents (no token embedding).
        self.global_encoder = TransformerEncoder(cfg)
        self.local_decoder = LocalByteDecoder(
            byte_dim=byte_dim, patch_size=patch_size, d_model=cfg.d_model,
            n_local_heads=n_local_heads, dropout=cfg.dropout,
            norm_layer=cfg.norm_layer(),
        )

    def forward(self, byte_ids: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        """byte_ids: (B, L) values 0..255 + BYTE_PAD; byte_mask: (B, L) True=real.

        Returns byte logits (B, L', 256) where L' = ceil(L/patch_size)*patch_size.
        """
        latents, patch_mask = self.local_encoder(byte_ids)
        memory = self.global_encoder(latents, mask=patch_mask, repr_input=True)
        return self.local_decoder(memory)

    @torch.no_grad()
    def predict_bytes(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> List[list]:
        """Greedy single-pass decode.  Returns per-sample predicted byte lists
        of exactly the requested (ground-truth) lengths."""
        self.eval()
        L = byte_ids.size(1)
        arange = torch.arange(L, device=byte_ids.device)
        mask = arange[None, :] < lengths[:, None]
        if byte_ids.is_cuda:
            major, minor = torch.cuda.get_device_capability(byte_ids.device.index)
            # bf16 on Ampere+, fp16 on older GPUs (e.g. 2080 Ti).
            amp_dtype = torch.bfloat16 if major * 10 + minor >= 80 else torch.float16
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = self.forward(byte_ids, mask)
        else:
            logits = self.forward(byte_ids, mask)
        pred = logits.argmax(dim=-1)  # (B, L')
        return [pred[i, : int(lengths[i].item())].cpu().tolist()
                for i in range(byte_ids.size(0))]
