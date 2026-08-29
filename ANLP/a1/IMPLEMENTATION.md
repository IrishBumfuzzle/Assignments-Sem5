# Implementation details

This file documents how the code in `src/` implements `ANLP_M26_A1.md`
(a from-scratch seq2seq Transformer for XOR-cipher decryption, five ablation
configs). Read together with `README.md` (runbook) and `Report.tex` (narrative).

## 1. Tokenization

### 1.1 Ciphertext side — learned BPE over a *phase-annotated* byte alphabet
- The ciphertext is the XOR of the plaintext with the 8-byte key `ANLP2026`, so
  byte `i` decrypts as `C[i] XOR K[i % 8]`.
- `src/dataset.py: PhaseByteBPE` (tokenizers library, byte-level) first maps
  each ciphertext byte to a unique *phase-annotated* byte:
  `symbol = (byte, i % 8)` → code `byte * 256 + (i % 8)`, a 2048-symbol base
  alphabet. The model never has to infer `i % 8` from context: each token
  carries its own phase.
- BPE then learns merges **over this alphabet** (standard learned subword
  tokenization, 0 UNK on all splits — the required "learned subword
  tokenization" for the ciphertext side).
- **Vocabulary size: 512 (official).** Only 424 of the 2048 phase-annotated
  symbols occur in the data, so a 512 vocab = 424 base symbols + 88 learned
  merges and stays within a few percent of 1:1 token↔byte (mean 508 tokens/line
  vs 213 at 8000). The 8000-vocab runs (kept in
  `outputs/ablations/vocab8000/`) show why the merge budget matters: multi-byte
  tokens break the 1:1 token/byte index correspondence that the cipher
  structure provides, and greedy decoding (the scored metric) plateaus at
  0.69 for all four autoregressive configs; the 512 vocab restores the
  correspondence and the same configs reach 0.9663–0.9996 bit accuracy.
  See Report.tex §2 ("Vocabulary size") and §3 (diagnosis of the 8000 plateau,
  including the prefix-sweep evidence).
- The token stream is **variable-length** (learned merges), not fixed-width
  byte chunks.

### 1.2 Target side — one token per plaintext byte (design choice)
- The first attempt tokenized plaintext with a second BPE vocab
  (ciphertext-BPE → plaintext-BPE). That collapsed to the English prior
  (0.668 bit acc ≈ 0.662 always-space) because two *independent* BPE
  segmentations of the same text disagree: the boundary overlap (IoU) between
  the two BPE token boundaries on matched lines is **0.41**. The model must
  learn an ordinal alignment between segmentations before it can decrypt.
- Fix: target = one token per plaintext **byte** (256-way alphabet + BOS/EOS).
  Target position `i` = source byte position `i`; the alignment is the identity
  and the cipher is a per-position, per-phase substitution. With a raw-byte
  (no-phase) BPE source the same architecture stalls at ~0.62; with the
  phase-annotated source it reaches 0.89–0.90 teacher-forced by epoch 28.
- Consequence: sequences are ~500–2670 target tokens. Training is possible
  because of activation checkpointing (§3) and the fp16 path (§3.1).
  Decoding at inference is O(T²) attention per line; the evaluation harness
  (`src/eval.py`) caps the target length at 1024 bytes for scoring.

## 2. Architecture (`src/models/`)

All five configs share: `dim=256`, `n_heads=8` (C3: `kv_heads=4`),
`n_layers=4`, `dim_ff=1024`, `dropout=0.1`, sinusoidal PE (C2: RoPE),
Pre-LayerNorm (C4: RMSNorm), learned positional-encoding-free local decoders
for C5. No `nn.Transformer` / `nn.MultiheadAttention` anywhere — every
component below is implemented from scratch in `src/models/`.

| file | what is implemented from scratch |
|------|----------------------------------|
| `positional.py` | `SinusoidalPE` (fixed) and `RoPE` (rotary, applied to Q/K) |
| `attention.py` | `MHA` + `GQA` (shared K/V head groups), causal mask, KV-cache for incremental decoding |
| `norm.py` | `LayerNorm` and `RMSNorm` (fp32 accumulators for fp16 safety) |
| `transformer.py` | encoder/decoder stacks, FFN, Pre-LN block layout, activation-checkpointed blocks |
| `blt.py` | C5: local patch encoder (banded 16-byte causal window, byte + rolling-hash n-gram embeddings, cross-attention pooling), global patch transformer, local patch decoder |
| `entropy_patching.py` | C5 *dynamic-patch variant* (explored, not official): `EntropyByteLM` auxiliary next-byte model, entropy-trigger boundary placement, fixed-stride reference, `patch_stats` |

### 2.1 C5: token-free, **fixed 4-byte patches (official)**
The ciphertext is consumed as **raw bytes** (no subword tokenization) — the
token-free BLT design. Consecutive bytes are grouped into fixed 4-byte patches
(BLT paper §2.1); the final patch of a line holds the remaining 1–4 bytes
(`fixed_stride` in `entropy_patching.py`). Per patch: a local transformer
encodes the bytes (byte embeddings + rolling-hash `n`-gram embeddings,
`n=3..8`, banded 16-byte causal window) and cross-attention pools them into one
latent; the global transformer (same 4/8/256 budget) runs over the patch
latents; a symmetric local decoder expands each latent back into 256-way byte
logits for the patch's positions. Decoding is **non-autoregressive** (one pass
over the 1:1 byte correspondence) — no prefix is fed back, so the exposure
problem of §4 does not exist by construction. The checkpoint saves
`patching = {method: "fixed", patch_size: 4}`.

### 2.2 C5 variant: entropy-based dynamic patching (explored, negative result)
Per the BLT paper §2.3 we also implemented entropy-driven patch boundaries: an
auxiliary `EntropyByteLM` (small 4-layer causal LM over raw bytes, 256-way,
no BOS) is trained first (~2 epochs), then patches are closed when the sum of
the next-byte entropy since the last boundary reaches a calibrated threshold
`θ_g` (calibrated so the mean patch is the target size 4.0 bytes; at
`θ_r=1.0` the calibration gives `θ_g≈4.63`, mean patch 4.0, max 12, ~436k
patches on train). The encoder/decoder handle variable patch lengths
(banded attention per actual length). A full 120-epoch run of this variant
scores **0.9931 bit / 0.0305 sequence accuracy** vs the fixed scheme's
**0.9999 / 0.9343** (Report.tex Table `tab:c5patch`): the dynamic boundaries
scatter single-byte errors and collapse exact-sequence accuracy, so the
**official C5 is the fixed-stride configuration** (`--patching fixed`, the
default). The dynamic variant remains fully supported via `--patching entropy`
(run `SKIP_FIXED=1 bash scripts/submit_c5.sh`); its results are in
`outputs/ablations/c5-entropy/`.

### 2.3 Autoregressive C1–C4
Standard encoder–decoder: sinusoidal (C2: RoPE) positional encoding,
multi-head (C3: grouped-query) cross-attention with causal self-attention in
the decoder, Pre-LayerNorm (C4: RMSNorm) blocks, learned source embeddings
(512×256) and a 256-way byte output head. The encoder is wrapped in
`transformer.py` blocks with activation checkpointing (§3).

## 3. Training

- `src/train.py` — single entry point; CLI args in §5.
- Optimizer: AdamW (`β=0.9/0.98`, weight decay `1e-2`). Schedule: 250-step
  linear warmup → cosine decay `5e-4 → 1e-5` over the whole epoch budget
  (40 epochs C1–C4; 120-epoch budget for C5).
- Effective batch 16 (C1–C4: batch 8 × grad-accum 2; C5: batch 16).
- fp16 autocast + `GradScaler` on Turing GPUs; bf16 without scaler on
  Ampere+ (auto-selected by compute capability).
- **Activation checkpointing** of all encoder/decoder blocks (numerically
  identical to the naive forward) makes the long byte targets fit 11 GB at
  batch 8.
- **Random batches, not length-bucketed.** With length-homogeneous buckets the
  alignment stalls (~0.67 bit acc by ep 6); identical data/schedule with
  shuffled random batches reaches ~0.83 by ep 6. Bucketed updates see one
  length band per step, so the model never consolidates a single alignment
  rule across lengths. The DataLoaders therefore shuffle all lines each
  epoch.
- The 8000-vocabulary runs (historical) also tried **random prefix dropout**
  (`--dropout-prefix-q`) and **scheduled sampling** (`--scheduled-sampling*`):
  both implemented, both failed (Report.tex §3, table `tab:fixes`). They
  remain available via CLI; the official study does not use them.

### 3.1 Results (official, greedy test, 512 vocabulary)

| Config | Bit acc | Seq acc | Levenshtein↓ | BLEU | ROUGE-L | Best ep |
|--------|---------|---------|--------------|------|---------|---------|
| C1 Base | **0.9996** | **0.9722** | 0.12 | **0.9998** | **0.9999** | 40 |
| C2 RoPE | 0.9663 | 0.7372 | 8.26 | 0.9909 | 0.9880 | 39 |
| C3 GQA | 0.9969 | 0.8932 | 0.34 | 0.9991 | 0.9997 | 40 |
| C4 RMSNorm | 0.9981 | 0.9615 | 0.22 | 0.9997 | 0.9999 | 40 |
| C5 fixed | **0.9999** | 0.9343 | **0.08** | 0.9996 | 0.9999 | 112/116 |
| (C5 entropy variant) | 0.9931 | 0.0305 | 14.0 | 0.9406 | 0.9716 | 76 |

Baseline: always-space bit acc 0.662. 8000-vocabulary ablation (same
architecture/schedule, 500 test lines): C1 0.6902/0.0060, C2 0.6773/0.0000,
C3 0.6905/0.0040, C4 0.6902/0.0020 — the tokenization lever is worth
≈0.29–0.31 bit acc, an order of magnitude larger than any architectural
effect (Report.tex table `tab:token`).

## 4. The greedy-gap diagnosis (8000 vocabulary — historical)

The 8000-vocab C1 runs plateaued at 0.69 greedy bit accuracy (0.927
teacher-forced). The diagnosis (full version in Report.tex §3):

- **Prefix sweep** (`/tmp/diag4.py`, 30 val lines, ep-25 checkpoint): oracle
  prefix → 0.893 (matches TF 0.898, so the incremental KV-cache path is
  exact); randomizing 10%/30%/50%/80%/100% of prefix positions → 0.791/0.722/
  0.688/0.664/0.654; re-anchoring to oracle every 2/5/10 bytes → 0.767/0.708/
  0.691. The decoder uses the *content* of its own prefix to locate the source
  position; one wrong byte and it never re-locates, then emits English prior.
- **Why prefix-corruption fixes failed**: random dropout teaches robustness to
  *obviously random* bytes; scheduled sampling corrupts with *plausible* bytes
  but the model learns to distrust even correct prefixes (TF also drops).
  Neither teaches re-localization from a wrong prefix.
- **Resolution**: the 8000 plateau was not pure exposure bias — it was the
  exposure failure *compounded by the multi-byte BPE resegmentation* (token
  index ≠ byte index). The 512 vocab (§1.1) removes the compounding: the
  TF–greedy gap at ep 10 shrinks from 0.184 to 0.016 and to ~0 by ep 40, and
  greedy no longer plateaus (C1: 0.9506 → 0.9937 → 1.0000 over ep 10/20/40).
  Residual exposure bias remains (cleanest in C2: TF 0.9998 vs greedy 0.9592,
  late-line repetition loops).

## 5. `train.py` CLI (official values in **bold**)

```
--config C1|C2|C3|C4|C5   --epochs **40** (C5: 120)   --batch-size **8** (C5: 16)
--lr **5e-4**  --dim **256**  --heads **8**  --kv-heads **4** (C3)
--layers **4** --dim-ff **1024**  --dropout **0.1**
--max-src **1024**  --max-tgt 512 (AR BPE target cap; with byte targets it is
      overridden to the longest plaintext in the splits, ≈2672)
--patch-size **4**  --byte-dim **64**   (C5 local enc/dec width)
--vocab-size **512** (C1–C4)
--grad-accum **2** (C5: 1)
--patching **fixed** | entropy        (C5 only; fixed = official)
--max-patch 32  --target-patch-size 4.0  --theta-r 1.0   (entropy variant)
--device cuda  --seed 42  --out outputs/<cfg>  --quick (10-line/3-epoch smoke test)
--wandb (default on; off in --quick)  --no-hf
```

Notes: `--patching` default is `fixed`. In the entropy variant, the auxiliary
entropy LM is trained first (its loss is logged as `entropy_lm/*`), the
threshold is calibrated to the target mean patch size, and patch statistics
(`patch_stats`: length histogram with exact float `hist_edges` + counts,
min/max/mean, trigger rate) are logged and saved in the checkpoint.
`--quick` smoke-tests the full pipeline (tokenizer → train → eval → artifacts)
on 10 lines × 3 epochs without WandB/HF.

## 6. Evaluation & metrics (`src/eval.py`, `src/utils.py`)

- **Teacher-forced eval** (per epoch, val set, best-checkpoint selection):
  bit accuracy, exact-sequence accuracy, Levenshtein, corpus BLEU (1–4 grams,
  brevity penalty), ROUGE-L (sentence F1 mean). C5 has no TF/greedy
  distinction (single non-autoregressive pass).
- **Greedy val eval** every 10 epochs (C1–C4): incremental KV-cache decoding
  with the model's own prefix — the same path the test uses.
- **Test eval** (once, best checkpoint, greedy, cap 1024 bytes for C1–C4
  scoring; C5 is single-pass): writes `results.json`, `test_samples.txt`
  (10 REF/PRED pairs), `training_curves.png` (loss + TF/greedy val bit acc,
  greyscale), copies `config.json`/`history.json`.
- The per-epoch eval reorders the references to the DataLoader's iteration
  order (loss is invariant; set-level metrics are not) — verified by
  confirming oracle incremental decoding reproduces the TF evaluation
  bit-for-bit (Report.tex §2, note (iii)).
- `scripts/eval_checkpoint.py <ckpt>` re-runs the full test evaluation for any
  saved checkpoint (used to assemble `outputs/C5` from the training log +
  `model_best.pt` via `scripts/make_c5_official.py`).

## 7. Ablation summary (official = 512 vocabulary)

| Axis | Finding (greedy test) |
|------|----------------------|
| Tokenization (8000→512 BPE) | **dominant lever**: +0.29–0.31 bit acc for all four AR configs (0.6773–0.6905 → 0.9663–0.9996); seq acc 0–0.6% → 73.7–97.2% |
| PE: sinusoidal → RoPE | C2 clearly worst (0.9663/0.7372; late-line repetition loops; 20-ep greedy stall even at 512 before escaping) — absolute index tracking beats relative-only for alignment-heavy tasks |
| Attention: MHA → GQA(4) | small cost (0.9969/0.8932 vs 0.9996/0.9722) for halved KV heads / −0.53M params |
| Norm: LayerNorm → RMSNorm | indistinguishable (0.9981/0.9615) |
| AR → non-AR token-free (C5) | best bit acc (0.9999) + best Levenshtein (0.08); C1 still best seq acc (0.9722); no exposure by construction |
| C5 patching: fixed → entropy-dynamic | strongly negative on seq acc (0.0305 vs 0.9343) — fixed stride is official |
| Target: BPE → bytes | 0.668 → 0.9996 (removes the resegmentation; §1.2) |
| Source alphabet: bytes → phase-annotated | 0.62 → 0.90 (25-ep probes) — phase is free information |

## 8. Submission layout (`scripts/make_submission.sh <roll>`)

The zip contains: `src/`, `scripts/`, `report/Report.tex`, `README.md`,
`IMPLEMENTATION.md`, `outputs/C1..C5/` (config, results, samples, curves —
**no** `.pt` files; checkpoints are on Hugging Face), and
`outputs/ablations/` (8000-vocab tokenization ablation `vocab8000/`, the
15-ep vocab probe `probe-c1-v512-ep15/`, the C5 dynamic-patch variant
`c5-entropy/`). WandB link, HF checkpoint links, and the roll number are
embedded in `Report.tex` §Reproducibility and `README.md`.

## 9. How to reproduce

**Fresh machine (SLURM / 2080 Ti class):**

```bash
bash scripts/setup_cluster.sh        # creates .venv_cluster (torch cu124, tokenizers, wandb)
bash scripts/submit_c14_vocab512.sh  # C1–C4 official: 40ep, vocab 512, 4 parallel SLURM jobs
SKIP_FIXED=1 bash scripts/submit_c5.sh   # optional: C5 dynamic-patch variant (explored)
bash scripts/submit_c5.sh           # C5 official: fixed 4-byte patches, 120-epoch budget
```

**Direct (any single GPU):**

```bash
bash scripts/run_experiment.sh C1 40 8 5e-4 256 8 4 1024 512 4 64 512 2 4
#  args: CONFIG EPOCHS BATCH LR DIM HEADS LAYERS MAX_SRC MAX_TGT PATCH BYTE_DIM VOCAB GRADACC KV
bash scripts/run_experiment.sh C5 120 16 5e-4 256 8 4 1024 512 4 64 8000 1 4 fixed
```

**Regenerate tables / zip:**

```bash
python scripts/make_results_table.py outputs
bash scripts/make_submission.sh 63237038   # → outputs/submission.zip
```

**WandB (project `irishbumfuzzle-team/anlp-assignment1`):** official runs C1
`d1kf0gih`, C2 `a00tbq4j`, C3 `ei6jhlnk`, C4 `4n9j9a0p`, C5 `ufvkwo3r`;
8000-vocab ablation C1 `pz4akfn6` / C2 `k7i8v6qx` / C3 `5ssmwo4e` / C4
`1j34vx49`; C5 dynamic-patch variant `bo1hepy2`. HF repos:
`IrishBumfuzzle/anlp-a1-C1..C5` (re-upload all five after the 512 retrain;
the dev machine has no HF token — upload from a machine that does, or set
`HF_TOKEN`).

## 10. File map

```
src/
  dataset.py            PhaseByteBPE (+ byte-phase alphabet), splits, DataLoaders, cipher helpers
  models/
    positional.py       SinusoidalPE, RoPE
    attention.py        MHA, GQA, causal mask, KV-cache
    norm.py             LayerNorm, RMSNorm (fp32 accumulators)
    transformer.py      C1–C4 encoder/decoder stacks, checkpointed blocks
    blt.py              C5 token-free model (fixed + variable patching paths)
    entropy_patching.py fixed_stride + entropy boundaries + EntropyByteLM (variant) + patch_stats
  eval.py               TF eval, greedy incremental decode, test-eval artifacts
  metrics.py            bit/seq/levenshtein/BLEU/ROUGE-L
  utils.py              scheduler, seeding, logging, wandb helpers (_log_patch_stats)
  train.py              CLI entry point (C1–C5, fp16/bf16, wandb, HF upload)
scripts/
  setup_cluster.sh      venv for the SLURM node
  run_experiment.sh     positional-arg wrapper for one config
  submit_c14_vocab512.sh / submit_c5.sh / run_all.sh   SLURM submission
  make_c5_official.py   assemble outputs/C5 (test eval + artifacts) from log + model_best.pt
  eval_checkpoint.py    re-run test eval for a saved checkpoint
  make_results_table.py regenerate the results tables from results.json files
  make_submission.sh    build the final zip
report/Report.tex       the report (compile to Report.pdf)
outputs/C1..C5/         per-config artifacts; outputs/ablations/ the ablation runs
```

## 11. Known issues / gotchas

- **Long targets + fp16**: LayerNorm/RMSNorm compute in fp32; GradScaler
  skips inf-norm steps. Do not switch to fp32 without cutting batch size.
- **Length bucketing**: never enable it; random batches are required for the
  alignment to learn (§3).
- **`--patching`**: default `fixed` (official C5). `entropy` requires the
  auxiliary LM phase (auto-run) and is the explored variant.
- **WandB histogram**: `patch_stats` saves exact float `hist_edges`
  (n+1 edges for n bins); `_log_patch_stats` has a fallback for old
  checkpoints that saved rounded `hist_centers` (a lossy format that once
  crashed `wandb.Histogram` on the server — fixed 2026-07-17).
- **C5 horizon**: the official fixed-patch run used a 120-epoch budget and was
  stopped at epoch 116 after the val loss was flat at ≈0.0010 for 25 epochs;
  best checkpoint is epoch 112. If you rerun, keep 120 epochs for the
  cosine-schedule shape.
- **HF upload** needs `HF_TOKEN` (dev machine has none; server runs use
  `SKIP_HF=1` and the user uploads manually).
- **Python 3.14**: `bytes.encode("latin1")` iteration yields ints — build
  byte lists explicitly (see `PhaseByteBPE`).
Did you put the model files in zip?
