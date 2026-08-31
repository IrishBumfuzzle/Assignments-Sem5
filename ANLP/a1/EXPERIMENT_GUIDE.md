# ANLP Assignment 1: Experiment & Architecture Guide

**Course:** Advanced Natural Language Processing (Spring 2026)  
**Topic:** Custom Transformers, XOR Cipher Decryption, Architectural Ablations (C1–C5), and Byte Latent Transformers (BLT)

---

## 1. Executive Summary & Cipher Analysis

### The Underlying Cipher
Analysis of `data/brown_cipher.txt` and `data/brown_plain.txt` reveals that the dataset is a **deterministic XOR stream cipher** using the repeating 8-byte ASCII key:

$$\text{Key} = \text{"ANLP2026"} \quad (\text{ASCII bytes: } [65, 78, 76, 80, 50, 48, 50, 54])$$

- **Encryption Formula**: $C[i] = P[i] \oplus K[i \pmod 8]$
- **Theoretical Optimum**: Because the mapping is 100% deterministic and information-preserving, a properly trained Seq2Seq model can achieve **~100% bit accuracy** and **~100% sequence accuracy**.

---

## 2. Failure Analysis & The "English Prior" Trap

In initial experiments, all models plateaued with **0% sequence accuracy** and characteristic bit accuracies:

| Configuration | Initial Bit Acc | Baseline Match | Behavior |
| :--- | :---: | :---: | :--- |
| **C1_Base** | `0.6564` | `0.6614` | Collapsed to predicting space / common English ASCII bits (`"the the..."`) |
| **C2_RoPE** | `0.6489` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C3_GQA** | `0.6591` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C4_RMSNorm** | `0.6522` | `0.6614` | Collapsed to predicting space / common English ASCII bits |
| **C5_BLT** | `0.5454` | `0.5470` | Collapsed to predicting empty / padding tokens |

---

## 3. Resolving the collapse (diagnosis trail)

1. **BPE↔BPE (subword target)**: the model converges (copy-like behaviour) but
   bit acc stays at the 0.668 baseline after 40 epochs — the target-side
   resegmentation (predicting BPE token *boundaries* of the plaintext) is the
   bottleneck, not the byte mapping. Dead end.
2. **Byte-level target** (target position i = plaintext byte i): removes
   resegmentation; a (byte, phase)-annotated cipher alphabet (symbol =
   `chr(b*8 + i mod 8)`, BPE over the 2048-symbol base, 8k vocab) makes each
   source token self-contained (`b XOR K[phase]`). With this the model reaches
   **teacher-forced bit acc 0.898** (25-epoch truncated probe, ep23-25).
3. **Random batches, not length bucketing**: bucketed batches stall alignment
   (0.67 at ep6 vs 0.83 for random batches on identical data/schedule).
4. **The teacher-forced / greedy gap (exposure bias)**: the epoch-25 model
   scores 0.898 teacher-forced but only **0.682 greedy** (all-space baseline
   0.662). Controlled prefix sweep (30 val lines):

   | decoder prefix fed | bit acc |
   | :--- | :---: |
   | oracle (ground truth) | 0.893 |
   | 10% of prefix randomised | 0.791 |
   | 50% randomised | 0.688 |
   | 100% randomised | 0.654 |
   | resync to oracle every 2 bytes | 0.767 |
   | pure greedy (own prefix) | 0.676 |

   The model stays aligned only while the fed prefix is correct; one early
   wrong byte drifts it permanently (it uses prefix *content* to locate the
   source position). The incremental decoder machinery is exact (oracle
   incremental == batch teacher-forcing, 0.893 vs 0.898).
5. **Random prefix dropout does NOT fix it** (15-epoch full-length probes):
   q=0.5 → tf 0.774 / greedy 0.679; q=1.0 → tf 0.662 / greedy 0.657. It
   teaches the model to ignore *obviously wrong* (random) bytes, but greedy
   failure is caused by *plausible* wrong bytes (fluent wrong English).
   q=1.0 removes the bootstrap signal entirely (the model falls back to the
   English prior).
6. **Scheduled sampling (the fix)**: every `--ss-every` epochs the whole
   training set is greedy-decoded with the current weights; each sample's
   predicted prefix is cached, and with probability ramping 0 →
   `--ss-max-p` over `--ss-ramp` epochs the sample is trained against its own
   cached prefix (supervision stays on the true target). Positions beyond
   `--ss-max-prefix-len` (default 1024) keep the true prefix. The decode pass
   uses a pre-allocated KV-cache buffer (in-place writes, no per-step
   `torch.cat` — the growing-cache concat is O(T²) memory traffic; verified
   bit-identical to the concat path).

### Useful training flags (C1–C4 byte targets)

```
--scheduled-sampling --ss-max-p 0.5 --ss-ramp 8 --ss-every 4 --ss-max-prefix-len 1024
--prefix-dropout 0.5     # random prefix dropout (diagnostic; see above)
--eval-greedy-every 3    # full greedy val eval every N epochs
--length-bucketing       # OFF by default; bucketing stalls alignment
--target-bpe             # BPE target (the collapsed regime, for ablation)
--no-phase-alphabet      # raw-byte BPE source (ablation)
```

### Timing (RTX 4050 6 GB, full lengths, B=2×accum 8, bf16)

- train epoch ≈ 155–240 s; greedy self-prefix pass (cap 1024, gen batch 32)
  ≈ 25 min for an untrained model (no EOS yet), ≈ 10–15 min once EOS is
  learned; greedy val eval ≈ 1 min. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

