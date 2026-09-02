# IMPLEMENTATION.md — ANLP M26 A1: XOR-Cipher Decryption with a From-Scratch Seq2Seq Transformer

This document describes the full implementation: architecture, tokenization, data
pipeline, training, evaluation, and — importantly — the experimental history
(every diagnostic that shaped the final design, with numbers) and the bugs found
and fixed along the way. Read `README.md` for the quick start and `report/Report.tex`
for the write-up.

---

## 1. Repository layout

```
src/
  dataset.py          data loading, BPE training/serialization (BPETextTokenizer,
                      PhaseByteBPE), datasets, collation, batch samplers
  models/
    attention.py      MHA + GQA (from scratch, scaled dot-product, KV cache),
                      FFN (GELU), pre-norm encoder/decoder blocks
    positional.py     sinusoidal PE + RoPE (from scratch)
    norm.py           LayerNorm + RMSNorm (from scratch, fp32 stats for fp16 safety)
    blt.py            C5 patch modules: LocalByteEncoder / LocalByteDecoder
    transformer.py    TransformerConfig, encoder/decoder stacks, Seq2SeqTransformer
                      (forward, generate() with KV cache), BLTModel (C5)
  train.py            config dispatch (CONFIGS C1-C5), training loop, AMP,
                      scheduling, evaluation, checkpoints, WandB, HF upload,
                      exposure-bias probes (prefix dropout, scheduled sampling)
  utils.py            metrics (bit acc, seq acc, Levenshtein, corpus BLEU,
                      ROUGE-L) — all vectorized NumPy, no NLTK — + plots
scripts/
  setup_cluster.sh    venv + deps (PyTorch cu124 wheel for the 2080 Ti)
  run_experiment.sh   single-GPU one-config runner (WandB + HF)
  submit_all.sh       SLURM job script + submission of all 5 configs
  eval_checkpoint.py  re-evaluate any checkpoint (greedy test pass)
  make_results_table.py  regenerate report tables from outputs/*/results.json
  make_submission.sh  build the Moodle zip (code + outputs, no .pt files)
report/Report.tex     the report
data/                 brown_cipher.txt (bit strings), brown_plain.txt (one per line)
outputs/<C>/          config.json, results.json, model_best.pt, model_last.pt,
                      test_samples.txt, training_curves.png, tokenizers/
```

Everything in `src/` is hand-written: no `nn.Transformer`, `nn.TransformerEncoder`,
or `nn.MultiheadAttention` anywhere. Only low-level primitives (`Linear`,
`Embedding`, `Dropout`) and `F.scaled_dot_product_attention` for the actual
attention matmul (the query/key/value construction, masking, GQA head grouping,
and KV caching are all ours).

## 2. Task and cipher

- 5000 Brown-corpus lines (min 21 B, median 554 B, p90 1086 B, max 2670 B).
- `C[i] = P[i] XOR K[i mod 8]`, key `ANLP2026` (8 bytes). Ciphertext ships as a
  bit string in `data/brown_cipher.txt`.
- Split 80/10/10 with seed 42: 4000 / 500 / 500.

## 3. Architecture (all from scratch)

Shared budget: `d_model=256`, 8 heads, 4 encoder + 4 decoder layers, FFN 1024
(4x), dropout 0.1, pre-LayerNorm (or pre-RMSNorm in C4).

### 3.1 Attention (`models/attention.py`)
- Standard scaled dot-product attention written out by hand:
  `scores = QK^T / sqrt(d_head)`, causal mask for decoder self-attention,
  source-length mask for cross-attention, softmax in fp32 under autocast
  (`scores.float()` → softmax → cast back).
- **GQA (C3)**: `n_kv_heads=4`; K/V projections produce 4 heads, then
  `repeat_interleave` up to 8 before the per-head attention; the head-grouping
  reshape is done with view/transpose, no copying beyond the repeat.
- **KV cache for inference**: `forward` accepts an optional
  `(k_buf, v_buf, write_pos)` tuple. New K/V are written *in place* into a
  pre-allocated buffer (`buf[:, :, wpos:wpos+Lq].copy_(k)`), and the next
  `write_pos` is returned. This replaces the naive per-step `torch.cat`
  (O(T²) total copies) with O(1) writes. Verified **bit-identical** to the
  cat-based path across 120 decode positions on a trained checkpoint
  (0 mismatches) before it was adopted.

### 3.2 Positional encodings (`models/positional.py`)
- **Sinusoidal**: fixed `sin/cos` table registered as a buffer (fp32, cast to
  the input dtype), added to embeddings.
- **RoPE (C2)**: `_cos_sin` computed in fp32 for all lengths up front; `apply`
  rotates q and k pairs in fp32 and casts back. Standard 2D rotation form
  (rotate-half / rotate-half, `θ_i = 10000^(-2i/d)`).

### 3.3 Normalization (`models/norm.py`)
- LayerNorm and RMSNorm written out (ε=1e-5, learnable affine).
- **fp16 safety**: statistics (mean/variance or RMS) and the affine
  multiplication are computed in fp32 (`x.float()`), then cast back to the
  input dtype. On a Turing GPU (fp16 autocast + GradScaler) this is essential
  — RMS in fp16 overflows/underflows at d=256. Under bf16/fp32 it is a
  numerical no-op.

### 3.4 Seq2Seq model (`models/transformer.py`)
- `forward(src, src_mask, tgt, tgt_mask, dec_input=None)`: encoder pass (or
  cached encoder output), decoder with causal mask, final linear to the target
  vocabulary. `dec_input` is the hook used by the exposure-bias probes
  (prefix dropout / scheduled sampling) to feed a *corrupted* decoder input
  while supervising against the true target.
- `generate(src, src_mask, max_len, bos_idx)`: greedy decode with the
  pre-allocated KV cache (§3.1). Allocates one `(k_buf, v_buf)` pair per
  decoder attention layer (`cache_dtype = autocast dtype if CUDA+autocast else
  param dtype`), tracks `write_pos` per layer, stops a row on its first EOS.
  **Verified correct**: incremental decode with the oracle prefix reproduces
  teacher-forced per-position logits exactly (0 diverged positions across 653
  positions) — i.e., there is no KV-cache or position-tracking bug in the
  decoding path.

### 3.5 C5 — BLT (`models/blt.py`)
Token-free, non-autoregressive:
- Source: raw ciphertext **bytes**, no tokenization (1 byte = 1 position,
  capped at `max_src_len=1024` bytes; lines longer than that are *excluded*,
  giving 4363/5000 lines: 3506/431/426 per split).
- Each byte is embedded into a `byte_dim=64` embedding (plus a learned
  patch-neighborhood context: `patch_size=4` — a local 1D convolution mixes the
  4-byte window, so the model sees adjacent bytes and the phase pattern).
- Target: the plaintext byte at the **same index** (1:1 mapping), predicted in
  parallel with a 256-way softmax. No autoregression ⇒ **no exposure bias**.
- This config is the "what if there were no alignment problem at all" baseline.

### 3.6 Activation checkpointing
All encoder and decoder blocks are wrapped with
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` during training.
Numerically identical to the uncheckpointed forward (verified), costs ~30% extra
compute, and is what lets 2672-position byte targets train at batch 8 on an
11 GB GPU (peak ≈ 7.2 GB).

## 4. Tokenization — the decisions that mattered most

### 4.1 Ciphertext side (assignment-mandated: learned BPE)
- **Byte-level BPE**, vocabulary 8000, `min_frequency=2`, learned on the
  training split (4000 lines) and frozen for val/test. No fixed-width 8-bit
  chunking anywhere.
- **Phase-annotated alphabet**: the symbol at byte index `i` is
  `(byte, i mod 8)` — 2048 base symbols, mapped to codepoints
  `b*8 + (i mod 8)` (so `chr(b*8 + i%8)`), and BPE learns merges over this
  alphabet. Rationale: with an 8-byte repeating key, byte `i` decrypts with
  `K[i mod 8]`; since the phase is a deterministic function of the index,
  folding it into the alphabet turns decryption into a per-symbol lookup
  `b XOR K[p]`. This was the single largest tokenization improvement measured
  (§8, probe D vs C).

### 4.2 Target side (our design choice)
- **One token per plaintext byte**: `tgt = [BOS] + list(plain.encode("latin1")) +
  [EOS]`, 259-way vocabulary. Position `i` in the target corresponds exactly to
  byte `i` of the plaintext and to source byte position `i`. The alignment the
  model must learn is the identity, and the cipher reduces to a per-position
  substitution.
- Why not a second BPE on the plaintext? See §8 (probe A): BPE↔BPE collapses to
  the English prior (0.668 bit acc after 40 epochs vs 0.662 always-space),
  because the model must first solve ordinal alignment between two *independent*
  segmentations. Measured boundary overlap of the two BPE tokenizations
  (intersection-over-union of cut positions on matched lines): **IoU 0.41**.
- Target length is set to `max(plain_len) + 2` across the splits
  (2672 for the full data; 800 in quick mode) so no line is truncated during
  the official runs.

### 4.3 BPE implementation (`dataset.py: BPETextTokenizer`, `PhaseByteBPE`)
Hand-written byte-level BPE: start from the base alphabet (256 bytes; for the
phase-annotated source, the 2048 `(byte, phase)` symbols), repeatedly count
adjacent-pair frequencies, merge the most frequent pair above `min_frequency`,
until the vocabulary cap (8000). Encoding replays the learned merge rules
(dict of pair→token, applied to completion). `PhaseByteBPE` wraps this over the
`b*8 + (i mod 8)` symbol mapping. Tokenizers serialize to JSON alongside
checkpoints so evaluation and inference are reproducible without retraining.

## 5. Data pipeline (`dataset.py`)

- `TokenizedCipherDataset` (BPE↔BPE, the collapsed baseline) and
  `ByteTargetCipherDataset` (official C1–C4 recipe): cipher bit string → bytes →
  phase-annotated BPE ids → src; plain → byte id list + BOS/EOS → tgt. Lines
  are **dropped** (not truncated) if `len(src) > max_src_len` or
  `len(tgt) > max_tgt_len`.
- `ByteCipherDataset` (C5): raw cipher bytes (≤1024), raw plain bytes, 1:1 aligned.
- Collation: right-pad src (mask), right-pad tgt (loss mask ignores PAD);
  `drop_last` only on train when requested (default off).
- **Random batches are the default.** `LengthBatchSampler` / `--length-bucketing`
  exist but are opt-in, because length-homogeneous batches measurably stall
  alignment learning (§8, finding 5).

## 6. Training (`train.py`)

- **Optimizer/schedule**: AdamW (β 0.9/0.98, wd 0.01), peak lr 5e-4, linear
  warmup 250 steps, cosine decay to 1e-5 over the remaining steps. Gradient
  clipping 1.0. Effective batch 16: batch 8 × grad-accum 2 (C1–C4), batch 16 × 1
  (C5).
- **AMP auto-selection** (`get_amp_settings()`): compute capability ≥ 8.0 →
  bf16 autocast, no scaler; cc < 8.0 (Turing/Volta, e.g. the 2080 Ti) → fp16
  autocast + `torch.cuda.amp.GradScaler`. The scaler path is
  `scale(loss/accum).backward()` → `unscale_(opt)` → `clip_grad_norm_` →
  `scaler.step(opt)` → `scaler.update()`. Validated on the 4050 in fp16 mode
  (forward/backward/scaler/generate, no NaNs) before the 2080 Ti run.
- **Per-epoch evaluation**: teacher-forced pass over the 500-line val set
  (bit acc, seq acc, Levenshtein, BLEU, ROUGE-L) — cheap, stable, and the right
  signal for long byte targets. **Full greedy decoding** on val every
  `--eval-greedy-every` epochs (10 in the official runs) — expensive (KV-cache
  decode over up to 2672 positions) but the metric the assignment scores.
  **Best checkpoint** = lowest val teacher-forced loss; both best and last are
  saved. Final test pass: greedy decoding on the 500 test lines with the best
  checkpoint.
- **Logging**: WandB (metrics per epoch + per-step loss, config, sample dumps;
  3-attempt connect with backoff, then automatic offline fallback so a network
  blip can never kill a multi-hour run). Best checkpoint uploaded to a
  per-config HF repo (`huggingface_hub`), when `HF_TOKEN` is set.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the run scripts
  (reduces fragmentation on long variable-length sequences).

## 7. Evaluation metrics (`utils.py`, all hand-written NumPy)

- **Bit accuracy** (the assignment's headline metric): each string is encoded
  UTF-8 → 8 bits/byte; prediction and reference are **aligned to the reference
  length** (short predictions zero-padded, long ones truncated), and accuracy is
  the fraction of matching bits. Every reference bit contributes exactly one
  comparison, so metrics are comparable across configs and line lengths.
- **Sequence accuracy**: exact full-string match, averaged.
- **Levenshtein**: classic edit distance (insert/delete/substitute),
  vectorized row recurrence (prefix-min trick, `np.minimum.accumulate`) — O(n·m)
  with no Python inner loop, which matters at 2672 × 2672.
- **BLEU**: corpus-level, **character** 1–4-gram precision with brevity
  penalty (standard form). Character-level because the outputs are byte strings
  and word segmentation of garbled output is not well-defined.
- **ROUGE-L**: sentence-level F1 over the **character** LCS, averaged.
- **Evaluation-alignment fix**: set-level metrics require the reference list to
  be in the same order the loader yields batches (`_ordered_refs` — gather refs
  by the loader's index tensor). Per-batch loss is unaffected by ordering,
  which is why this bug was invisible in training curves and only showed up as
  "BLEU/seq-acc nonsense" at eval time.

## 8. Experimental history — how the design was found

This is the diagnostic trail; every number below is from a real run (C1 unless
stated). It is also the substance of the report's Design Decisions section.

1. **Baseline BPE↔BPE (probe A).** Ciphertext BPE → plaintext BPE, the obvious
   first design. After 40 epochs: **0.668 bit acc** on val greedy — essentially
   the **0.662 always-space baseline** (predicting every byte as `0x20`).
   Teacher-forced loss kept decreasing, so the model was fitting the *marginal*
   distribution of English, not the line-specific mapping. Copy controls
   (model asked to copy its input; model asked to decrypt a *constant 1-byte*
   key) confirmed the 4-layer budget is consumed by ordinal alignment between
   the two segmentations, not by the cipher. Boundary IoU of the two BPE
   tokenizations: **0.41**.
2. **Byte target + raw-byte BPE source (probe C).** Removes resegmentation on
   the target side; source BPE over plain bytes (no phase). 25-epoch probe:
   **~0.62** — better than collapse, but the model still has to compute
   `i mod 8` from position itself.
3. **Phase-annotated source (probe D).** `(byte, i mod 8)` alphabet, BPE over
   it. 25-epoch probe: **0.89–0.90 teacher-forced by epoch 25** (vs ~0.62 for
   the raw-byte source). Full-length 40-epoch official run: 0.892 at ep15 →
   0.9197 at ep25 → **0.927 at ep40**. This is the final source alphabet.
4. **Length bucketing stalls alignment (finding 5).** With length-homogeneous
   buckets (the "standard" efficiency trick), C1 stalls at **~0.67 by epoch 6**;
   identical data/schedule with **random** batches reaches **~0.83 by epoch 6**.
   Hypothesis: each bucketed step sees a single length band, so the identity
   alignment rule is never consolidated across lengths in one step. Random
   batches became the default (`--length-bucketing` opt-in).
5. **Full-length, no truncation.** With random batches the model needs the full
   2672-position targets to consolidate; truncating the target (e.g. to 1024)
   caps TF accuracy around 0.90. Official runs use full lengths
   (activation checkpointing makes it fit).
6. **The exposure-bias diagnosis (the centerpiece).** At 40 epochs C1 reaches
   **0.927 TF bit acc but 0.690 greedy** (test). A 30-line val sweep on the
   epoch-25 checkpoint controls *what the decoder is fed as its own prefix*:
   oracle prefix → 0.893 (matches TF 0.898 ⇒ decoding machinery is exact);
   10/30/50/80/100% of prefix positions randomized → 0.791/0.722/0.688/0.664/
   0.654 (100% random is *below* the space baseline); resync to oracle every
   k=2/5/10 bytes → 0.767/0.708/0.691. The model uses prefix *content* to index
   the source position; one wrong greedy byte ⇒ permanent drift ⇒ output falls
   back to the English LM prior (fluent but wrong).
7. **Three failed mitigations (all 15-epoch probes, full lengths):**
   - **Prefix dropout q=0.5** (replace half the fed prefix ids with random ids
     in training): TF 0.774, greedy 0.679 — hurts TF, no greedy gain.
   - **Prefix dropout q=1.0**: TF 0.662, greedy 0.657 — no bootstrap signal,
     collapses to the English prior.
   - **Scheduled sampling** (every 4 epochs, greedy-decode the train set with
     the current weights, cache prefixes; feed a sample its own prefix with
     probability ramping 0→0.5 over 8 epochs; prefix length capped at 1024
     bytes): TF 0.814, greedy 0.664 — even *below* the baseline greedy.
   Common thread: corrupting any tested fraction of the decoder input makes the
   model distrust a *correct* prefix (TF drops) without teaching it to
   re-locate the source from a wrong one. The model learns to ignore *obviously
   random* bytes, but greedy failure comes from *plausible* wrong bytes.
   **Decision: official runs use plain teacher forcing** (best TF *and* best
   greedy of everything tested); the diagnosis + failed mitigations are the
   report's analytical contribution. (Implementation for all three remains in
   `train.py`: `--prefix-dropout`, `--scheduled-sampling`, `--ss-max-p`,
   `--ss-ramp`, `--ss-every`, `--ss-max-prefix-len`.)
8. **C5 as the attribution control.** The non-autoregressive 1:1-byte model
   hits **1.0000 bit acc / 0.9343 exact sequences / Levenshtein 0.1** with 3.4M
   params in 15 minutes. Same cipher, same data — the only difference is that
   there is no autoregressive prefix to drift. This proves C1–C4's 0.69 is an
   *alignment-maintenance* failure under greedy decoding, not a cipher-learning
   failure.

## 9. Final results (official runs, 40 epochs, single RTX 2080 Ti, fp16+scaler)

| Config | Params | Bit acc | Seq acc | Levenshtein | BLEU | ROUGE-L | Best ep | Time |
|--------|--------|---------|---------|-------------|------|---------|---------|------|
| C1 Base | 9.54M | 0.6902 | 0.0060 | 313.8 | 0.6870 | 0.6908 | 35 | 134 min |
| C2 RoPE | 9.54M | 0.6773 | 0.0000 | 720.2 | 0.1840 | 0.3922 | 18 | 150 min |
| C3 GQA (kv=4) | 9.02M | 0.6905 | 0.0040 | 311.2 | 0.6921 | 0.6945 | 36 | 141 min |
| C4 RMSNorm | 9.54M | 0.6902 | 0.0020 | 321.4 | 0.6804 | 0.6832 | 36 | 132 min |
| C5 BLT | 3.39M | 1.0000 | 0.9343 | 0.1 | 0.9996 | 0.9998 | 39 | 15 min |

Test set: 500 lines (C5: 426 — lines with >1024-byte ciphertext are excluded
from its raw-byte source). Baseline: always-space 0.662. TF val bit acc at best
epoch: C1 0.9268, C2 0.8754, C3 0.9255, C4 0.9246, C5 0.9999.

Interpretation (full version in the report): GQA and RMSNorm are
quality-neutral; RoPE is clearly worse (absolute alignment over ~1000 positions
needs absolute position identity, which RoPE's relative encoding forces the
model to reconstruct from a possibly-wrong prefix — it plateaus at 0.875 TF and
degenerate-repeats under greedy); C5 shows what the cipher itself is worth when
the alignment is given for free.

## 10. Bug log (found and fixed)

| Bug | Symptom | Fix |
|-----|---------|-----|
| Reference ordering in eval | BLEU/seq-acc garbage while loss curves fine | `_ordered_refs`: permute refs to loader iteration order |
| Length bucketing | TF bit acc stuck ~0.67 (ep6) vs 0.83 random | random batches default; `--length-bucketing` opt-in |
| Per-batch vs per-bit metric confusion in probes | wrong "fix worked" conclusions | metrics centralized in `utils.py`, probes call the same functions |
| Naive KV cache O(T²) cat per step | greedy eval ~16 min/epoch | pre-allocated buffers + in-place `copy_`; verified bit-identical |
| Norms in fp16 on Turing | NaN/instability on 2080 Ti | LayerNorm/RMSNorm compute stats in fp32 |
| RoPE cos/sin in fp16 | drift at long positions | `_cos_sin` in fp32 |
| Attention softmax in fp16 | -inf/NaN on long masks | softmax in fp32 under autocast |
| Wandb API key typo in an old script | instant `CommError` | key from `~/.netrc`; plus 3-attempt retry + offline fallback in `train.py` |
| `pgrep -f "A\|B"` | ERE: `\|` is literal; double-launched a run, OOM | use `|` or separate patterns; kill by explicit PID |

## 11. Reproduction

```bash
bash scripts/setup_cluster.sh                 # venv with torch (cu124) + deps
export WANDB_API_KEY=...                       # wandb team irishbumfuzzle-team
export HF_TOKEN=...                            # optional; uploads best ckpts
bash scripts/run_experiment.sh C1              # one config, ~2.2 h on 2080 Ti
bash scripts/submit_all.sh                     # SLURM: all five in parallel
python scripts/eval_checkpoint.py --checkpoint outputs/C1/model_best.pt
python scripts/make_results_table.py outputs   # regenerate report tables
bash scripts/make_submission.sh <roll_number>  # the Moodle zip
```

Single-GPU sanity check before a long run:
`PYTHONPATH=. .venv/bin/python src/train.py --config C1 --output-dir
/tmp/quick --epochs 1 --max-line-bytes 120` (finishes in a couple of minutes).

## 12. Known limitations

- Single seed per config (spread within ±0.014 across C1/C3/C4; ordering of the
  ablation is unaffected at this scale).
- Greedy-only evaluation (assignment protocol); beam/sampling not scored.
- C5 excludes lines longer than the 1024-byte raw source cap.
- RoPE's negative result is task-specific (alignment-heavy, long, positional
  correspondence) and should not be generalized to semantic seq2seq.
- The byte-level target is a target-side design choice; the ciphertext
  tokenization is learned BPE as the assignment requires.
