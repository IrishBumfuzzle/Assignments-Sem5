"""Evaluation metrics, greedy decoding, and training utilities."""

from collections import Counter
import math
import random
from typing import Dict, List, Optional, Union
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def causal_mask(length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a lower-triangular boolean causal mask of shape [length, length].

    True indicates allowed attention positions, False indicates future masked positions.
    """
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


def text_to_bits(text: str) -> str:
    """Convert text string into UTF-8 binary bitstring ('0' and '1')."""
    return "".join(f"{b:08b}" for b in text.encode("utf-8", errors="replace"))


def bit_level_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute the percentage of exact bit matches between prediction and target."""
    if not targets:
        return 0.0

    total_acc = 0.0
    for pred, tgt in zip(predictions, targets):
        pred_bits = text_to_bits(pred)
        tgt_bits = text_to_bits(tgt)

        max_len = max(len(pred_bits), len(tgt_bits))
        if max_len == 0:
            total_acc += 1.0
            continue

        # Pad shorter bitstring with zeros for comparison
        pred_padded = pred_bits.ljust(max_len, "0")
        tgt_padded = tgt_bits.ljust(max_len, "0")

        matches = sum(p == t for p, t in zip(pred_padded, tgt_padded))
        total_acc += matches / max_len

    return total_acc / len(targets)


def sequence_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Compute the percentage of sequences that are perfectly reconstructed (100% exact match)."""
    if not targets:
        return 0.0
    return sum(1.0 for p, t in zip(predictions, targets) if p == t) / len(targets)


def levenshtein_distance(prediction: str, target: str) -> int:
    """Compute minimum edit distance between prediction and target strings."""
    m, n = len(prediction), len(target)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if prediction[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + 1)
            prev = temp

    return dp[n]


def compute_bleu(predictions: List[str], targets: List[str], max_order: int = 4) -> float:
    """Compute corpus BLEU score (BLEU-4) from scratch with brevity penalty."""
    if not targets:
        return 0.0

    p_ns = [0.0] * max_order
    total_clipped = [0] * max_order
    total_candidates = [0] * max_order
    total_cand_len = 0
    total_ref_len = 0

    for pred, tgt in zip(predictions, targets):
        cand_tokens = pred.strip().split()
        ref_tokens = tgt.strip().split()

        total_cand_len += len(cand_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_order + 1):
            cand_ngrams = Counter(
                [tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)]
            )
            ref_ngrams = Counter(
                [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
            )

            clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
            total_clipped[n - 1] += clipped
            total_candidates[n - 1] += max(0, len(cand_tokens) - n + 1)

    for n in range(max_order):
        if total_candidates[n] > 0:
            p_ns[n] = max(1e-8, total_clipped[n] / total_candidates[n])
        else:
            p_ns[n] = 1e-8

    # Brevity penalty
    if total_cand_len == 0:
        return 0.0
    if total_cand_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len / total_cand_len)

    # Geometric mean of modified n-gram precisions
    log_p_sum = sum(math.log(p) for p in p_ns) / max_order
    return bp * math.exp(log_p_sum)


def compute_rouge(predictions: List[str], targets: List[str]) -> float:
    """Compute average ROUGE-L (Longest Common Subsequence) F1 score."""
    if not targets:
        return 0.0

    f1_scores = []
    for pred, tgt in zip(predictions, targets):
        cand_tokens = pred.strip().split()
        ref_tokens = tgt.strip().split()

        m, n = len(cand_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            f1_scores.append(1.0 if m == n else 0.0)
            continue

        # LCS length using DP
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        prec = lcs_len / m
        rec = lcs_len / n

        if prec + rec > 0:
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def compute_metrics(
    predictions: List[str], targets: List[str], tokenized: bool = True
) -> Dict[str, float]:
    """Compute all evaluation metrics specified in the assignment."""
    if not predictions or not targets:
        return {
            "bit_accuracy": 0.0,
            "sequence_accuracy": 0.0,
            "levenshtein": 0.0,
            "bleu": 0.0 if tokenized else None,
            "rouge": 0.0 if tokenized else None,
        }

    distances = [levenshtein_distance(p, t) for p, t in zip(predictions, targets)]
    metrics = {
        "bit_accuracy": bit_level_accuracy(predictions, targets),
        "sequence_accuracy": sequence_accuracy(predictions, targets),
        "levenshtein": sum(distances) / max(1, len(distances)),
    }

    if tokenized:
        metrics["bleu"] = compute_bleu(predictions, targets)
        metrics["rouge"] = compute_rouge(predictions, targets)
    else:
        metrics["bleu"] = None
        metrics["rouge"] = None

    return metrics


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    source: torch.Tensor,
    tokenizer: object,
    max_len: int = 512,
    source_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Autoregressive greedy decoding for sequence generation."""
    model.eval()
    if device is None:
        device = source.device

    source = source.to(device)
    if source_mask is not None:
        source_mask = source_mask.to(device)

    bos_id = getattr(tokenizer, "bos_id", 1)
    eos_id = getattr(tokenizer, "eos_id", 2)
    b = source.size(0)

    # Encode source sequence
    memory = model.encode(source, src_mask=source_mask)

    generated = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = causal_mask(generated.size(1), device=device)
        logits = model.decode(generated, memory, tgt_mask=tgt_mask, src_mask=source_mask)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

        # If a sequence already finished, keep padding or eos
        next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, eos_id), next_tokens)
        generated = torch.cat([generated, next_tokens], dim=1)

        finished = finished | (next_tokens.squeeze(1) == eos_id)
        if finished.all():
            break

    return generated
