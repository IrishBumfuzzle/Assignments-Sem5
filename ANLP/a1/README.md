# Assignment 1: Custom Transformers, Architectural Variants, and Byte Latent Transformers (BLT)
**Course:** Advanced Natural Language Processing (Spring 2026)

---

## 1. Project Overview

This repository contains a full Seq2Seq Transformer built **from scratch using basic PyTorch operations** (without `nn.Transformer` or `nn.MultiheadAttention`) to learn mappings from encrypted binary cipher sequences (`brown_cipher.txt`) to plaintext English sentences (`brown_plain.txt`).

It includes implementations of 5 controlled ablation configurations:
- **C1 (Base)**: Sinusoidal Absolute Positional Encoding + Multi-Head Attention (MHA) + Pre-LayerNorm + Subword BPE (8k vocab).
- **C2 (RoPE)**: Rotary Positional Embedding (RoPE) + Multi-Head Attention (MHA) + Pre-LayerNorm + Subword BPE (8k vocab).
- **C3 (GQA)**: Sinusoidal Absolute Positional Encoding + Grouped-Query Attention (GQA, `kv_heads=4`) + Pre-LayerNorm + Subword BPE (8k vocab).
- **C4 (RMSNorm)**: Sinusoidal Absolute Positional Encoding + Multi-Head Attention (MHA) + RMSNorm + Subword BPE (8k vocab).
- **C5 (BLT Token-Free)**: Sinusoidal Absolute Positional Encoding + Multi-Head Attention (MHA) + Pre-LayerNorm + Byte Latent Transformer (Local Encoder/Decoder).

For an in-depth technical analysis of the XOR cipher, the English prior failure mode, and empirical results, refer to [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

---

## 2. Directory Structure

```
<rollnumber>_assignment1/
|-- src/
|   |-- models/
|   |   |-- attention.py     # SDPA, MHA, GQA, and FFN modules from scratch
|   |   |-- positional.py    # Sinusoidal PE and RoPE modules from scratch
|   |   |-- norm.py          # Custom LayerNorm and RMSNorm modules from scratch
|   |   `-- blt.py           # Local Encoder and Local Decoder for BLT
|   |-- dataset.py           # SubwordTokenizer (BPE 8k) & ByteTokenizer DataLoaders
|   |-- train.py             # Main training, evaluation, AMP, DataParallel & WandB loop
|   `-- utils.py             # Metrics (Bit Acc, Seq Acc, Levenshtein, BLEU, ROUGE) & Greedy Decode
|-- scripts/
|   |-- run_experiment.sh    # SLURM experiment runner
|   `-- submit_all.sh        # Batch job submission for C1-C5
|-- outputs/                 # Checkpoints, logs, and evaluation outputs
|-- data/                    # brown_cipher.txt & brown_plain.txt
|-- EXPERIMENT_GUIDE.md      # Comprehensive post-mortem & experimental guide
`-- README.md                # Setup instructions, documentation & links
```

---

## 3. Setup and Installation

### Prerequisites
- Python >= 3.10 (tested on Python 3.14)
- [uv](https://github.com/astral-sh/uv) or standard `pip`

### Install Dependencies
```bash
# Using uv:
uv sync

# Or using pip:
pip install torch numpy tokenizers wandb
```

---

## 4. Running Experiments

### On SLURM Cluster:
```bash
# Submit all 5 configurations
bash scripts/submit_all.sh

# Or submit individual configuration:
sbatch --job-name=anlp_C1 --gres=gpu:2 scripts/run_experiment.sh C1 15 16 5e-4 256 8 4 1024 512 4 64 8000 1
```

### Locally:
```bash
# Configuration 1: Base Transformer
uv run python src/train.py --config C1 --epochs 15 --batch-size 16 --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 --vocab-size 8000 --wandb

# Configuration 2: Rotary Positional Encodings (RoPE)
uv run python src/train.py --config C2 --epochs 15 --batch-size 16 --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 --vocab-size 8000 --wandb

# Configuration 3: Grouped-Query Attention (GQA)
uv run python src/train.py --config C3 --epochs 15 --batch-size 16 --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 --vocab-size 8000 --wandb

# Configuration 4: RMSNorm Normalization
uv run python src/train.py --config C4 --epochs 15 --batch-size 16 --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 --vocab-size 8000 --wandb

# Configuration 5: Byte Latent Transformer (BLT Token-Free)
uv run python src/train.py --config C5 --epochs 15 --batch-size 16 --lr 5e-4 --dim 256 --heads 8 --layers 4 --max-src-len 1024 --max-tgt-len 512 --patch-size 4 --byte-dim 64 --wandb
```

---

## 5. Evaluation Metrics

During validation and testing, the model evaluates greedy autoregressive decoding on:
- **Bit-Level Accuracy**: Percentage of exact bit matches between decoded text and ground truth UTF-8 binary representations.
- **Sequence Accuracy**: Percentage of sentences reconstructed with 100% exact match.
- **Levenshtein Distance**: Minimum edit distance between predictions and ground truth.
- **BLEU & ROUGE-L Scores**: Standard n-gram overlap and longest common subsequence overlap metrics.
