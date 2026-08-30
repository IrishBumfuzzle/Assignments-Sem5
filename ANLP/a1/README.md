# ANLP M26 — Assignment 1: Transformer Ablation Study (from scratch)

Learned decryption of a repeating-key XOR cipher: a Seq2Seq Transformer built
**from scratch** (no `nn.Transformer` / `nn.MultiheadAttention`) maps the
bit-string ciphertext in `data/brown_cipher.txt` to the plaintext in
`data/brown_plain.txt`.

The cipher is `C[i] = P[i] XOR K[i mod 8]` with key `"ANLP2026"` (verified on
all 5000 lines). The theoretical optimum is 100% bit/sequence accuracy, so the
test set is a pure model-quality probe.

## Ablation configs

| Config | Positional encoding | Attention   | Normalization | Tokenization            |
|--------|--------------------|-------------|---------------|-------------------------|
| C1     | Sinusoidal         | MHA         | LayerNorm     | BPE subword (8k vocab)  |
| C2     | RoPE               | MHA         | LayerNorm     | BPE subword (8k vocab)  |
| C3     | Sinusoidal         | GQA (kv=4)  | LayerNorm     | BPE subword (8k vocab)  |
| C4     | Sinusoidal         | MHA         | RMSNorm       | BPE subword (8k vocab)  |
| C5     | Sinusoidal         | MHA         | LayerNorm     | BLT token-free (raw bytes, patch 4) |

Shared: `dim=256, heads=8, layers=4, dim_ff=1024, dropout=0.1, lr=5e-4
(warmup 250 + cosine to 1e-5), AdamW (wd 0.01), grad clip 1.0, effective
batch 16 (8 x 2 accumulation), 15 epochs, max_src 1024, max_tgt 512, seed 42`.

BPE tokenizers (one for cipher bit-strings, one for plaintext) are **learned
subword** tokenizers trained on the train split only (8:1:1 split). The
ciphertext is tokenized over its raw bit characters — learned merges, *not*
fixed 8-bit chunks.

## Repository layout

```
|-- src/
|   |-- models/
|   |   |-- attention.py   # SDPA, MHA, GQA, FFN, encoder/decoder blocks
|   |   |-- blt.py         # BLT local byte encoder/decoder (C5)
|   |   |-- norm.py        # LayerNorm, RMSNorm (from scratch)
|   |   |-- positional.py  # Sinusoidal PE, RoPE (from scratch)
|   |   `-- transformer.py # Seq2SeqTransformer (C1-C4), BLTModel (C5)
|   |-- dataset.py         # BPE tokenizers, tokenized & byte loaders
|   |-- train.py           # Main training loop with WandB
|   `-- utils.py           # Metrics (bit acc, seq acc, Levenshtein, BLEU, ROUGE-L) + plots
|-- scripts/
|   |-- run_experiment.sh  # SLURM job script (any config)
|   |-- submit_all.sh      # Submit C1..C5
|   `-- setup_cluster.sh   # One-time .venv_cluster setup
|-- data/                  # brown_cipher.txt, brown_plain.txt
|-- outputs/               # Per-config results, checkpoints, plots, tokenizers
`-- Report.pdf             # Final report
```

## Setup

```bash
uv sync                      # creates .venv with torch + deps
# or manually:
pip install torch tokenizers numpy matplotlib wandb huggingface_hub
```

## Running

Quick smoke test (256-sample subset, 2 epochs, ~1 min per config):

```bash
python src/train.py --config C1 --quick
```

Full single config (WandB logging + HF upload):

```bash
WANDB_API_KEY=... HF_TOKEN=... \
python src/train.py --config C1 --wandb --hf-repo <user>/anlp-a1-c1
```

SLURM (cluster):

```bash
bash scripts/setup_cluster.sh        # once, on a node
bash scripts/submit_all.sh           # submits C1..C5 (15 epochs each)
# single config: sbatch --job-name=anlp_C2 scripts/run_experiment.sh C2
# with HF upload:  HF_REPO=<user>/anlp-a1-c2 sbatch ... scripts/run_experiment.sh C2
```

Notes:
- Mixed precision is selected automatically: **bf16** on Ampere+ GPUs,
  **fp16 + GradScaler** on older ones (e.g. RTX 2080 Ti, compute cap 7.5).
- Batch 8 x 2 accumulation = effective 16, the same effective batch the
  reference setup uses; bump `--batch-size`/`--eval-batch-size` on bigger
  GPUs for throughput (results unchanged).
- Expected runtime: ~30–60 min per config on a 2080 Ti; C5 (BLT) is much
  faster (single-pass non-autoregressive decode).
- C5 skips BLEU/ROUGE by design (token-free; the assignment restricts those
  metrics to the tokenized models C1-C4).

## Outputs (per config, in `outputs/<C>/`)

- `results.json` — final test metrics + full training history
- `model_best.pt` / `model_last.pt` — checkpoints (best by val loss)
- `training_curves.png` — loss / bit acc / seq acc / Levenshtein / BLEU / ROUGE-L
- `test_samples.txt`, `val_samples_epoch1.txt` — reference/prediction samples
- `config.json`, `tokenizers/` — exact config + learned BPE tokenizers

Metrics: bit accuracy (byte-aligned, shorter predictions zero-padded to the
reference length), sequence accuracy (exact match), mean Levenshtein distance,
corpus BLEU with brevity penalty and mean sentence ROUGE-L F1 (char n-grams,
C1-C4 only). All metrics are greedy-decoded.

## Results

*(filled in after training runs — WandB run URLs and HuggingFace checkpoint
links go here and in the report.)*
