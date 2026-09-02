# ANLP M26 — Assignment 1: Transformer Ablation Study (from scratch)

Learned decryption of a repeating-key XOR cipher: a Seq2Seq Transformer built
**from scratch** (no `nn.Transformer` / `nn.MultiheadAttention`) maps the
bit-string ciphertext in `data/brown_cipher.txt` to the plaintext in
`data/brown_plain.txt`.

The cipher is `C[i] = P[i] XOR K[i mod 8]` with key `"ANLP2026"` (verified on
all 5000 lines). The theoretical optimum is 100% bit/sequence accuracy, so the
test set is a pure model-quality probe.

## Ablation configs

| Config | Positional encoding | Attention     | Normalization | Tokenization (C1–C4: BPE subword, 8k vocab) |
|--------|--------------------|---------------|---------------|---------------------------------------------|
| C1     | Sinusoidal         | MHA           | LayerNorm     | + byte-level target                          |
| C2     | RoPE               | MHA           | LayerNorm     | + byte-level target                          |
| C3     | Sinusoidal         | GQA (kv=4)    | LayerNorm     | + byte-level target                          |
| C4     | Sinusoidal         | MHA           | RMSNorm       | + byte-level target                          |
| C5     | Sinusoidal         | MHA           | LayerNorm     | BLT token-free (raw bytes, patch 4)          |

Shared: `dim=256, heads=8, layers=4, dim_ff=1024, dropout=0.1, lr=5e-4
(warmup 250 + cosine to 1e-5), AdamW (wd 0.01), grad clip 1.0, effective
batch 16 (8 x 2 accumulation on 11 GB GPUs; 4 x 4 or 2 x 8 on 6 GB cards),
40 epochs, max_src 1024 tokens, seed 42`.

## Tokenization / target design (C1–C4)

**Ciphertext side (required by the assignment: learned subword).**
Each ciphertext byte is represented by the symbol `(byte value, key position)`,
i.e. byte `b` at index `i` becomes the symbol `b*8 + (i mod 8)` (a
2048-symbol alphabet). The key position is a deterministic function of the
byte index — no extra information — and it turns decryption of a symbol into
the per-symbol lookup `b XOR K[i mod 8]`. **BPE is then learned over this
alphabet** (8000 = 2048 base symbols + ~5950 merges, `min_frequency=2`),
yielding variable-length tokens (~2.9 bytes/token, ~190 tokens/line median) —
a learned subword scheme, not fixed-width 8-bit chunking.

**Target side (our design choice: byte-level, 256-way output).** Target
position `i` is exactly plaintext byte `i` (no second BPE vocabulary). This
gives the model a 1:1 positional correspondence between target positions and
cipher bytes, so the decryption has no resegmentation component: the model
only has to align target position `i` to the source token containing cipher
byte `i` and apply the (now per-symbol) XOR lookup.

Why not a second BPE on the plaintext? We measured the segmentation
misalignment: the cipher-BPE and plain-BPE boundaries agree on only **41 %**
of positions (IoU), with a 1.7 : 1 token-count ratio, i.e. a true
cross-segmentation translation. Control experiments showed the failure mode:
(a) *copy* (src token `t` -> tgt token `t`) converges to near-zero loss, so a
4-layer transformer **can** learn ordinal cross-attention alignment when the
two sides share a segmentation; (b) the fully tokenized setup
(cipher-BPE -> plain-BPE) instead collapses to the English prior — bit acc
stuck at the 0.662 always-space baseline with val NLL *above* the marginal
prior after 40 epochs (train loss 0.72 = pure memorization). The bottleneck
is composing byte-offset accumulation on both sides with XOR phase tracking
and resegmentation inside 4 layers. Byte targets remove two of those four
compositions (target-side accumulation + resegmentation); annotating the
cipher symbols with the key phase removes the third (phase tracking).

Ablations of this design (same training budget, run as `C1-targetbpe` /
`C1-nophase`, in `outputs_abl/`):
- `--target-bpe` — BPE target (the fully tokenized setup above);
- `--no-phase-alphabet` — byte target with BPE over raw cipher bytes (no
  phase annotation, model must track `i mod 8` itself).

Long byte targets (up to ~2670 positions) are trained with **activation
checkpointing** (numerically identical, ~30% extra compute) and **random
batches**, so the same code runs on 6 GB and 11 GB GPUs. We initially used
length-homogeneous (bucketed) batches to reduce padding waste, but observed
that bucketing stalls alignment learning (bit acc ~0.67 at ep 6 vs ~0.83 for
random batches on identical data/schedule): with bucketing, each optimizer
step sees a single length band, and the model never consolidates one
cross-sequence alignment rule across lengths. Random batching is the default
(`--length-bucketing` re-enables the old behaviour).

One more metric pitfall we hit and fixed: with a non-sequential batch sampler
the evaluation references must be permuted to the loader's iteration order
(`_ordered_refs` in `train.py`); comparing predictions in sampler order
against references in dataset order silently corrupts every bit/Levenshtein/
BLEU number (the training loss is unaffected, since it is computed per
batch).

**C5 (BLT)** uses raw cipher bytes with patch size 4 and a token-free
local-attention encoder/decoder, exactly as in the assignment table.

## Repository layout

```
|-- src/
|   |-- models/
|   |   |-- attention.py   # SDPA, MHA, GQA, FFN, encoder/decoder blocks
|   |   |-- blt.py         # BLT local byte encoder/decoder (C5)
|   |   |-- norm.py        # LayerNorm, RMSNorm (from scratch)
|   |   |-- positional.py  # Sinusoidal PE, RoPE (from scratch)
|   |   `-- transformer.py # Seq2SeqTransformer (C1-C4), BLTModel (C5)
|   |-- dataset.py         # BPE tokenizers, datasets, collation (random batches default)
|   |-- train.py           # Main training loop with WandB + HF upload
|   `-- utils.py           # Metrics (bit acc, seq acc, Levenshtein, BLEU, ROUGE-L) + plots
|-- scripts/
|   |-- run_all.sh         # Run C1..C5 sequentially on one standalone GPU (target machine)
|   |-- run_experiment.sh  # SLURM job script (any config)
|   |-- submit_all.sh      # Submit C1..C5 (SLURM)
|   |-- eval_checkpoint.py # Re-evaluate a saved checkpoint (any split)
|   |-- make_results_table.py  # Print report tables from results.json files
|   |-- upload_to_hf.py        # (Re)upload checkpoints to HuggingFace
|   `-- setup_cluster.sh   # One-time .venv_cluster setup
|-- data/                  # brown_cipher.txt, brown_plain.txt
|-- outputs/               # Per-config results, checkpoints, plots, tokenizers
`-- Report.pdf             # Final report
```

## Setup

Recommended (what `scripts/run_all.sh` uses first):

```bash
bash scripts/setup_cluster.sh    # creates .venv_cluster with torch cu124 + deps
```

(Use `uv sync` / `pip install` only if the machine's NVIDIA driver supports
the newest torch CUDA wheels — the dev machine ran torch 2.13+cu130; the
cu124 wheels from `setup_cluster.sh` work with driver >= 550.)

## Running

Quick smoke test (256-sample subset, 2 epochs, ~2 min per config):

```bash
python src/train.py --config C1 --quick
```

**All 5 configs on one standalone GPU (RTX 2080 Ti, the target machine):**

```bash
bash scripts/setup_cluster.sh        # once: creates .venv_cluster
export HF_TOKEN=hf_...               # optional: checkpoint upload to HuggingFace
bash scripts/run_all.sh              # C1..C5, 40 epochs each (~13-15 h total)
```

`run_all.sh` uses the final recipe: C1–C4 with `--batch-size 8
--grad-accum-steps 2` (byte targets, T up to ~2672), C5 with `--batch-size 16
--grad-accum-steps 1`; fp16 + GradScaler is selected automatically on the
2080 Ti (cc 7.5). The official runs used the same recipe via `submit_all.sh`
(5 parallel SLURM jobs). WandB logs go to `irishbumfuzzle-team/anlp-assignment1`;
best checkpoints upload to `IrishBumfuzzle/anlp-a1-C{1..5}` when `HF_TOKEN`
is set (without it the runs still complete; upload the `.pt` files manually
later if needed).

Single config with WandB + HF upload (any machine):

```bash
WANDB_API_KEY=... HF_TOKEN=... \
python src/train.py --config C1 --wandb --hf-repo <user>/anlp-a1-c1
```

SLURM cluster (how the official runs were submitted):
```bash
bash scripts/setup_cluster.sh        # once, on a node
sbatch scripts/submit_all.sh         # 5 parallel jobs, one config each
sbatch scripts/run_experiment.sh C1  # or a single config
```

  **fp16 + GradScaler** on older ones (e.g. RTX 2080 Ti, compute cap 7.5).
- Byte targets have heavy length variance (median ~600 bytes, max ~2670).
  Training uses **random batches** (see the design note above); on a 6 GB
  card use `--batch-size 2 --grad-accum-steps 8`; on 11 GB
  `--batch-size 8 --grad-accum-steps 2` (the defaults in
  `scripts/run_experiment.sh`). Effective batch size is 16 either way.
- Validation: byte-target models are tracked per epoch with a fast
  **teacher-forced** pass (one forward + argmax); full **greedy** decoding is
  run every `--eval-greedy-every` epochs (default 10) and on the final test
  evaluation, per the assignment's greedy-decoding protocol. Best checkpoint
  is selected on val loss.
- BLEU / ROUGE-L (character n-grams) are computed for **all** configs for
  completeness; the assignment requires them for the tokenized configs C1–C4.
- Expected runtime: ~2–4 h per C1–C4 config (40 epochs, byte targets) on a
  2080 Ti; C5 is faster (single-pass non-autoregressive decode).

## Outputs (per config, in `outputs/<C>/`)

- `results.json` — final test metrics + full training history
- `model_best.pt` / `model_last.pt` — checkpoints (best by val loss)
- `training_curves.png` — loss / bit acc / seq acc / Levenshtein / BLEU / ROUGE-L
- `test_samples.txt`, `val_samples_epoch1.txt` — reference/prediction samples
- `config.json`, `tokenizers/` — exact config + learned BPE tokenizers

Metrics: bit accuracy (shorter predictions zero-padded to the reference
length), sequence accuracy (exact match), mean Levenshtein distance, corpus
BLEU with brevity penalty and mean sentence ROUGE-L F1 (char n-grams). All
reported metrics are greedy-decoded.

## The teacher-forced / greedy gap (exposure bias)

The byte-target model learns the decryption well under teacher forcing
(→ 0.90 bit acc, val), but the assignment's evaluation is **greedy**
decoding, where it stalls near **0.68** — barely above the always-space
baseline of 0.662. A controlled prefix sweep (see report, §
"Autoregressive decoding") shows the model stays source-aligned only while
the decoder prefix it was fed is correct: with the oracle prefix it matches
teacher forcing (0.893), each randomised fraction of the prefix costs
accuracy monotonically, and a fully random prefix drops it *below* the
space baseline. One early wrong greedy byte drifts the alignment permanently.

Three training-side mitigations were probed (15-epoch full-length runs,
C1) and **none** lifted greedy above the baseline:

| recipe | TF bit acc | greedy bit acc |
|---|---|---|
| baseline teacher forcing | ~0.87 | 0.68 |
| random prefix dropout q=0.5 | 0.774 | 0.679 |
| random prefix dropout q=1.0 | 0.662 | 0.657 |
| scheduled sampling p→0.5 (own greedy prefixes) | 0.814 | 0.664 |

Corrupting any tested fraction of the decoder input makes the model
distrust a *correct* prefix too (TF accuracy drops) without it ever learning
to re-locate the source position from a wrong one. The official runs
therefore use plain teacher forcing, and the report documents the diagnosis
and the failed mitigations in detail. The corresponding CLI flags exist for
reproducibility: `--prefix-dropout`, `--scheduled-sampling` (+ `--ss-max-p`,
`--ss-ramp`, `--ss-every`, `--ss-max-prefix-len`).

## Results

Official runs: 40 epochs, single RTX 2080 Ti (fp16 + GradScaler), 500-line
test set (C5: 426 — lines with >1024-byte ciphertexts are excluded from its
raw-byte source). Greedy decoding, best checkpoint by val loss.
Regenerate: `python scripts/make_results_table.py outputs`.

| Config | Bit acc | Seq acc | Levenshtein ↓ | BLEU | ROUGE-L | Best epoch |
|--------|---------|---------|---------------|------|---------|------------|
| C1 Base | 0.6902 | 0.0060 | 313.8 | 0.6870 | 0.6908 | 35 |
| C2 RoPE | 0.6773 | 0.0000 | 720.2 | 0.1840 | 0.3922 | 18 |
| C3 GQA (kv=4) | 0.6905 | 0.0040 | 311.2 | 0.6921 | 0.6945 | 36 |
| C4 RMSNorm | 0.6902 | 0.0020 | 321.4 | 0.6804 | 0.6832 | 36 |
| C5 BLT | **1.0000** | **0.9343** | **0.1** | 0.9996 | 0.9998 | 39 |

Baseline: always-space predictor 0.662 bit acc. Teacher-forced val bit acc at
best epoch (C1 0.927, C2 0.875, C3 0.926, C4 0.925, C5 0.9999) shows the
cipher itself is learned; the C1–C4 test scores are the exposure-bias ceiling
under greedy decoding (see `report/Report.tex`, Section 3, and
`IMPLEMENTATION.md` §8).

WandB: <https://wandb.ai/irishbumfuzzle-team/anlp-assignment1> (runs: C1
`pz4akfn6`, C2 `k7i8v6qx`, C3 `5ssmwo4e`, C4 `1j34vx49`, C5 `ldihh8vu`).
Hugging Face (best + last checkpoints, tokenizers, results):
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C1> –
<https://huggingface.co/IrishBumfuzzle/anlp-a1-C5>.
