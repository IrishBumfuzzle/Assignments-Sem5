"""Entropy-based dynamic patching for the C5 (BLT) configuration.

A scaled-down, faithful implementation of the entropy patching scheme from
"Byte Latent Transformer: Patches Scale Better Than Tokens" (Pagnoni et al.,
arXiv:2412.09871, Section 2.3, 4.2-4.4):

1. A small byte-level autoregressive LM (the *entropy model*) is trained on
   the TRAIN split only.  In our seq2seq setting the byte stream is the
   *source* (cipher) stream: the patch structure is a function of the source
   alone, so it is available identically at training and inference time with
   no target leakage.
2. Next-byte entropies H(x_i) are computed for every byte of every line
   under the entropy-LM distribution, with the context **reset at every line
   start** (paper Section 4.4, avoids entropy drift).
3. A new patch starts at position t > 0 when either
       H(x_t) > theta_g                          (global constraint)
       or  H(x_t) - H(x_{t-1}) > theta_r         (approx-monotonicity constraint)
   and every patch is clamped to at most ``max_patch`` bytes.
4. ``theta_g`` is calibrated on the training lines so that the *mean* patch
   size hits ``target_patch_size`` (default 4.0) -- this makes the dynamic
   scheme compute-matched to fixed stride-4 patching (paper Section 4.3).

The resulting patching function satisfies *incremental patching*
(f_p(x_<i) = f_p(x)_<i): the boundary decision at position t depends only on
bytes with index < t, so the same segmentation is obtained from a prefix of a
line as from the full line restricted to that prefix.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset import cipher_bytes_of, BYTE_PAD
from models.attention import TransformerEncoderLayer
from models.norm import LayerNorm

N_BYTES = 256
START_ID = 256  # start-of-line token for the entropy LM


@dataclass
class EntropyLMConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    window: int = 256      # sliding-window causal context in bytes
    max_len: int = 1025    # longest line (1024) + START token
    dropout: float = 0.0


class ByteEntropyLM(nn.Module):
    """Tiny autoregressive byte-level LM used to estimate next-byte entropies."""

    def __init__(self, cfg: EntropyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(N_BYTES + 1, cfg.d_model, padding_idx=START_ID)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, None, cfg.d_ff, cfg.dropout, LayerNorm
            )
            for _ in range(cfg.n_layers)
        )
        self.norm = LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, N_BYTES)
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _window_causal_mask(self, L: int, device) -> torch.Tensor:
        key = (L, device)
        m = self._mask_cache.get(key)
        if m is None:
            i = torch.arange(L)
            ok = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.cfg.window)
            m = ok.to(device)
            self._mask_cache[key] = m
        return m

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) values 0..256 (256 = START).  Returns logits (B, L, 256)."""
        L = ids.size(1)
        assert L <= self.cfg.max_len, f"line too long for entropy LM: {L}"
        pos = torch.arange(L, device=ids.device)
        x = self.embed(ids) + self.pos_embed(pos)[None]
        for layer in self.layers:
            x, _ = layer(x, mask=self._window_causal_mask(L, ids.device))
        return self.head(self.norm(x))


# --------------------------------------------------------------------------- #
# Training the entropy model                                                  #
# --------------------------------------------------------------------------- #
def train_entropy_lm(
    lines: List[Sequence[int]],
    cfg: EntropyLMConfig,
    epochs: int = 40,
    lr: float = 3e-4,
    batch_lines: int = 32,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    log_prefix: str = "[entropy]",
) -> ByteEntropyLM:
    """Train the entropy LM on the (train-split) byte lines.  Deterministic
    for a fixed seed.  Returns the model on CPU."""
    g = torch.Generator().manual_seed(seed)
    model = ByteEntropyLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tensors = [
        torch.tensor([START_ID] + [int(b) for b in line], dtype=torch.long)
        for line in lines
    ]
    losses = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(tensors), generator=g).tolist()
        ep_loss, n = 0.0, 0
        s = 0
        while s < len(order):
            # batch size from the next 16 lines (a superset of the actual
            # batch, so the (B, H, L, L) matrix stays bounded ~16K tokens)
            Lmax_rough = max(tensors[i].size(0) for i in order[s : s + 16])
            b = max(1, min(batch_lines, 16, 16384 // max(Lmax_rough, 1)))
            idxs = order[s : s + b]
            s += b
            Lmax = max(tensors[i].size(0) for i in idxs)
            x = torch.full((len(idxs), Lmax), BYTE_PAD, dtype=torch.long)
            for r, i in enumerate(idxs):
                t = tensors[i]
                x[r, : t.size(0)] = t
            x = x.to(device)
            # input position t (0 = START) predicts byte t of the line
            tgt = x[:, 1:]                       # (B, Lmax-1)
            logits = model(x)[:, : tgt.size(1)]  # (B, Lmax-1, 256)
            loss = F.cross_entropy(
                logits.reshape(-1, N_BYTES), tgt.reshape(-1), ignore_index=BYTE_PAD
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n += 1
        avg = ep_loss / max(n, 1)
        losses.append(avg)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{log_prefix} epoch {epoch:02d}/{epochs}  loss {avg:.4f}")
    return model.cpu()


# --------------------------------------------------------------------------- #
# Entropies and patch boundaries                                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def next_byte_entropies(
    lm: ByteEntropyLM,
    lines: List[Sequence[int]],
    device: torch.device = torch.device("cpu"),
    batch: int = 64,
) -> List[np.ndarray]:
    """Per-position next-byte entropies H(x_i) in bits for every line, with
    the LM context reset at each line start (each line is scored standalone,
    preceded by the START token)."""
    lm.to(device).eval()
    out: List[np.ndarray] = []
    for s in range(0, len(lines), batch):
        chunk = lines[s : s + batch]
        Lmax = max(len(x) for x in chunk)
        x = torch.full((len(chunk), Lmax + 1), START_ID, dtype=torch.long)
        for r, line in enumerate(chunk):
            x[r, 1 : len(line) + 1] = torch.tensor(list(line), dtype=torch.long)
        x = x.to(device)
        p = torch.softmax(lm(x)[:, : Lmax].float(), dim=-1)  # (B, Lmax, 256)
        H = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1).cpu()  # (B, Lmax)
        for r, line in enumerate(chunk):
            out.append(H[r, : len(line)].numpy())
    return out


def patch_line(
    H: np.ndarray,
    theta_g: float,
    theta_r: float,
    max_patch: int,
) -> Tuple[List[int], List[int]]:
    """Patch starts/lengths for one line given its next-byte entropies.

    Boundary at t (t >= 1) iff  H[t] > theta_g  or  H[t] - H[t-1] > theta_r
    or  the current patch has reached ``max_patch`` bytes.
    """
    L = H.shape[0]
    starts = [0]
    last = 0
    for t in range(1, L):
        if (H[t] > theta_g) or ((H[t] - H[t - 1]) > theta_r) or (t - last) >= max_patch:
            starts.append(t)
            last = t
    lens = [b - a for a, b in zip(starts, starts[1:])] + [L - starts[-1]]
    return starts, lens


def fixed_stride(L: int, P: int) -> Tuple[List[int], List[int]]:
    """Fixed strided patching (paper Section 2.1) -- the C5 ablation control."""
    starts = list(range(0, L, P))
    lens = [
        min(P, L - a) for a in starts
    ]
    return starts, lens


def mean_patch_size(lines_lens: List[Tuple[List[int], List[int]]],
                    line_lengths: List[int]) -> float:
    total_patches = sum(len(l) for _, l in lines_lens)
    total_bytes = sum(line_lengths)
    return total_bytes / max(total_patches, 1)


def _total_patches(entropies: List[np.ndarray], theta_g: float, theta_r: float,
                   max_patch: int) -> int:
    total = 0
    for H in entropies:
        L = H.shape[0]
        n = 1
        last = 0
        for t in range(1, L):
            if (H[t] > theta_g) or ((H[t] - H[t - 1]) > theta_r) or (t - last) >= max_patch:
                n += 1
                last = t
        total += n
    return total


def calibrate_theta_g(
    entropies: List[np.ndarray],
    theta_r: float,
    max_patch: int,
    target_avg: float,
    iters: int = 16,
) -> float:
    """Find the global threshold whose mean patch size on these lines matches
    ``target_avg`` (paper Section 4.3: "estimate a patching threshold that
    achieves a desired average patch size on the pretraining data mix").

    Mean patch size decreases monotonically in theta_g, so bisection applies.
    """
    line_lengths = [H.shape[0] for H in entropies]
    hi = max(H.max() for H in entropies) + 1.0   # no global triggers at all
    lo = -1.0                                     # every position a boundary
    total_hi = _total_patches(entropies, hi, theta_r, max_patch)
    mean_hi = sum(line_lengths) / total_hi
    total_lo = _total_patches(entropies, lo, theta_r, max_patch)
    mean_lo = sum(line_lengths) / total_lo
    if target_avg <= mean_lo:
        return lo
    if target_avg >= mean_hi:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        total_mid = _total_patches(entropies, mid, theta_r, max_patch)
        mean_mid = sum(line_lengths) / total_mid
        if mean_mid < target_avg:  # patches too large -> raise the floor
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def patch_stats(line_lengths: List[int],
                structures: List[Tuple[List[int], List[int]]]) -> dict:
    lens = np.concatenate([np.array(l, dtype=np.float64) for _, l in structures]) \
        if structures else np.zeros(0)
    total_bytes = sum(line_lengths)
    hist, centers = np.histogram(lens, bins=np.arange(0.5, lens.max() + 1.5)) \
        if lens.size else (np.zeros(1), np.array([1.0]))
    return {
        "n_patches": int(lens.size),
        "n_bytes": int(total_bytes),
        "mean_patch_size": float(lens.mean()) if lens.size else 0.0,
        "median_patch_size": float(np.median(lens)) if lens.size else 0.0,
        "min_patch_size": int(lens.min()) if lens.size else 0,
        "max_patch_size": int(lens.max()) if lens.size else 0,
        "std_patch_size": float(lens.std()) if lens.size else 0.0,
        "single_byte_patch_frac": float((lens == 1).mean()) if lens.size else 0.0,
        "hist_counts": [int(c) for c in hist],
        "hist_centers": [int(round(e)) for e in centers],
    }


def build_patch_structures(
    lines: List[Sequence[int]],
    theta_g: float,
    theta_r: float,
    max_patch: int,
    entropies: List[np.ndarray],
) -> List[Tuple[List[int], List[int]]]:
    return [patch_line(H, theta_g, theta_r, max_patch) for H in entropies]


def verify_incremental() -> bool:
    """Sanity check: patching a prefix must equal patching the full line
    restricted to that prefix (the paper's incremental-patching property)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = int(rng.integers(2, 60))
        H = rng.uniform(0.2, 3.0, size=L)
        theta_g, theta_r, max_patch = float(rng.uniform(0.5, 2.5)), float(rng.uniform(0.0, 0.6)), int(rng.integers(3, 9))
        s_full, _ = patch_line(H, theta_g, theta_r, max_patch)
        for k in range(1, L):
            s_pre, l_pre = patch_line(H[:k], theta_g, theta_r, max_patch)
            if any(b >= k for b in s_pre):
                return False
            if s_pre != [b for b in s_full if b < k]:
                return False
    return True
