"""Evaluation metrics (bit accuracy, sequence accuracy, Levenshtein, BLEU,
ROUGE-L) and plotting helpers.  All metrics are implemented from scratch.
"""

import math
from collections import Counter
from typing import List

import numpy as np


# --- string / bit helpers -------------------------------------------------------
def _bytes_of(text: str) -> bytes:
    return text.encode("utf-8")


def _bits_of(text: str) -> np.ndarray:
    b = _bytes_of(text)
    if len(b) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


# --- sequence accuracy -----------------------------------------------------------
def sequence_accuracy(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    return float(np.mean([1.0 if p == r else 0.0 for p, r in zip(preds, refs)]))


# --- bit-level accuracy -----------------------------------------------------------
def bit_accuracy(preds: List[str], refs: List[str]) -> float:
    """Fraction of matching bits, aligned to the reference length.

    Predictions shorter than the reference are zero-padded; longer ones are
    truncated, so every reference bit contributes exactly one comparison.
    """
    correct = total = 0
    for p, r in zip(preds, refs):
        rb = _bits_of(r)
        pb = _bits_of(p)
        n = len(rb)
        if n == 0:
            continue
        if len(pb) < n:
            pb = np.concatenate([pb, np.zeros(n - len(pb), dtype=np.uint8)])
        correct += int((pb[:n] == rb).sum())
        total += n
    return correct / total if total else 0.0


# --- Levenshtein -------------------------------------------------------------------
def _levenshtein_naive(a: str, b: str) -> int:
    """Reference O(mn) DP (used to validate the vectorized version)."""
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        new = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            new[j] = min(dp[j] + 1, new[j - 1] + 1, dp[j - 1] + (ca != cb))
        dp = new
    return dp[-1]


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute), O(len(a)*len(b)).

    Vectorized row recurrence.  For a row with `prev` fixed, define
    ``a[j] = min(prev[j]+1, prev[j-1]+cost[j])``; then
    ``cur[j] = min(a[j], a[j-1]+1, a[j-2]+2, ...)`` =
    ``min(a[j], j + min_{i<j}(a[i]-i))``, and the inner term is a prefix
    minimum (``np.minimum.accumulate``) - no Python inner loop.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(b) > len(a):
        a, b = b, a
    n = len(b)
    cb = np.array([ord(c) for c in b], dtype=np.int32)
    j_idx = np.arange(1, n + 1, dtype=np.int32)
    prev = np.arange(n + 1, dtype=np.int32)
    INF = np.iinfo(np.int32).max // 4
    for c in (ord(ch) for ch in a):
        cost = (c != cb)  # (n,) bool
        a_col = np.minimum(prev[1:] + 1, prev[:-1] + cost)  # delete / substitute
        g = a_col - j_idx
        prefix_min = np.minimum.accumulate(g)  # m[j] = min_{i<=j} g[i]
        shifted = np.empty(n, dtype=np.int32)
        shifted[0] = INF
        shifted[1:] = prefix_min[:-1]
        cur = np.empty(n + 1, dtype=np.int32)
        cur[0] = prev[0] + 1
        cur[1:] = np.minimum(a_col, j_idx + shifted)  # insert scan folded in
        prev = cur
    return int(prev[-1])


def mean_levenshtein(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    return float(np.mean([levenshtein(p, r) for p, r in zip(preds, refs)]))


# --- BLEU (corpus-level, character n-grams) -------------------------------------------
def _ngrams(s: str, n: int) -> List[str]:
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def corpus_bleu(refs: List[str], hyps: List[str], max_n: int = 4) -> float:
    """Standard corpus BLEU with brevity penalty (character n-grams)."""
    if not refs:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        num = den = 0
        for r, h in zip(refs, hyps):
            ref_counts = Counter(_ngrams(r, n))
            hyp_counts = Counter(_ngrams(h, n))
            num += sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
            den += sum(hyp_counts.values())
        if den == 0 or num == 0:
            return 0.0
        precisions.append(math.log(num / den))
    log_avg = sum(precisions) / len(precisions)
    len_ref = sum(len(r) for r in refs)
    len_hyp = sum(len(h) for h in hyps)
    bp = 1.0 if len_hyp > len_ref else math.exp(1 - len_ref / len_hyp) if len_hyp else 0.0
    return math.exp(min(0.0, log_avg)) * bp if precisions else 0.0


# --- ROUGE-L (sentence-level F1, averaged) ------------------------------------------------
def _lcs_len(a: str, b: str) -> int:
    """LCS length, vectorized: cur[j] = max(a'[j], max_{i<j} cur[i]) where
    a'[j] = max(prev[j], prev[j-1]+match[j]); the max-prefix is a running
    maximum, so each row is a single np.maximum.accumulate."""
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    cb = np.array([ord(c) for c in b], dtype=np.int32)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for c in (ord(ch) for ch in a):
        match = (c == cb)
        cur = np.empty(len(b) + 1, dtype=np.int32)
        cur[0] = 0
        cur[1:] = np.maximum.accumulate(
            np.maximum(prev[1:], prev[:-1] + match)
        )
        prev = cur
    return int(prev[-1])


def rouge_l_f1(ref: str, hyp: str) -> float:
    l = _lcs_len(ref, hyp)
    if l == 0:
        return 0.0
    p = l / len(hyp) if hyp else 0.0
    r = l / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean_rouge_l(refs: List[str], hyps: List[str]) -> float:
    if not refs:
        return 0.0
    return float(np.mean([rouge_l_f1(r, h) for r, h in zip(refs, hyps)]))


# --- aggregate ---------------------------------------------------------------------------
def compute_metrics(refs: List[str], hyps: List[str], tokenized: bool = True) -> dict:
    m = {
        "bit_accuracy": bit_accuracy(hyps, refs),
        "sequence_accuracy": sequence_accuracy(hyps, refs),
        "levenshtein": mean_levenshtein(hyps, refs),
    }
    if tokenized:
        # BLEU / ROUGE are reported for the tokenized configs (C1-C4).
        m["bleu"] = corpus_bleu(refs, hyps)
        m["rouge_l"] = mean_rouge_l(refs, hyps)
    return m


# --- plotting ------------------------------------------------------------------------------
def plot_training_curves(history: dict, save_path: str) -> None:
    """history: {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_bit_accuracy": [...],
    "val_sequence_accuracy": [...], "val_levenshtein": [...], "val_bleu": [...],
    "val_rouge_l": [...], "train_samples_per_sec": [...], "peak_mem_mb": [...]}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[utils] matplotlib not installed; skipping plot")
        return

    ep = history["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training curves", fontsize=14)

    axes[0, 0].plot(ep, history["train_loss"], label="train")
    axes[0, 0].plot(ep, history["val_loss"], label="val")
    axes[0, 0].set_title("Cross-entropy loss")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, history["val_bit_accuracy"], marker="o")
    axes[0, 1].set_title("Val bit accuracy"); axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(ep, history["val_sequence_accuracy"], marker="o")
    axes[0, 2].set_title("Val sequence accuracy"); axes[0, 2].set_xlabel("epoch")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(ep, history["val_levenshtein"], marker="o")
    axes[1, 0].set_title("Val mean Levenshtein"); axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(alpha=0.3)

    if "val_bleu" in history and history["val_bleu"] and history["val_bleu"][0] is not None:
        axes[1, 1].plot(ep, [v if v is not None else 0 for v in history["val_bleu"]], marker="o", label="BLEU")
        axes[1, 1].plot(ep, [v if v is not None else 0 for v in history["val_rouge_l"]], marker="s", label="ROUGE-L")
        axes[1, 1].legend()
    axes[1, 1].set_title("Val BLEU / ROUGE-L"); axes[1, 1].set_xlabel("epoch"); axes[1, 1].grid(alpha=0.3)

    if "peak_mem_mb" in history and history["peak_mem_mb"]:
        axes[1, 2].plot(ep, history["peak_mem_mb"], marker="o", color="green")
        axes[1, 2].set_title("Peak GPU memory (MB)"); axes[1, 2].set_xlabel("epoch")
        axes[1, 2].grid(alpha=0.3)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] saved plot to {save_path}")
