#!/bin/bash
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/%x_%j.log

# Location-independent: works from any clone of this repo.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR" || exit 1

CONFIG=${1:-C1}
EPOCHS=${2:-40}
BATCH_SIZE=${3:-8}
LR=${4:-5e-4}
DIM=${5:-256}
HEADS=${6:-8}
LAYERS=${7:-4}
MAX_SRC=${8:-1024}
MAX_TGT=${9:-512}
PATCH_SIZE=${10:-4}
BYTE_DIM=${11:-64}
VOCAB_SIZE=${12:-8000}
GRAD_ACCUM=${13:-2}
KV_HEADS=${14:-4}

echo "=========================================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $HOSTNAME"
echo "Repo:     $DIR"
echo "Start:    $(date)"
echo "Config:   $CONFIG | Epochs: $EPOCHS | Batch: $BATCH_SIZE (x$GRAD_ACCUM accum) | LR: $LR | Dim: $DIM | Heads: $HEADS (kv: $KV_HEADS) | Layers: $LAYERS | MaxSrc: $MAX_SRC | MaxTgt: $MAX_TGT | Vocab: $VOCAB_SIZE"
echo "=========================================="

nvidia-smi || true
mkdir -p /scratch/$USER/tmp 2>/dev/null || true
export TMPDIR=/scratch/$USER/tmp

# Virtual env: prefer .venv_cluster (SLURM nodes), fall back to .venv
if [ -f ".venv_cluster/bin/activate" ]; then
    source .venv_cluster/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: no .venv_cluster or .venv found in $DIR"; exit 1
fi

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-irishbumfuzzle-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-anlp-assignment1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_ARGS=""
if [ -n "$HF_REPO" ]; then
    HF_ARGS="--hf-repo $HF_REPO"
    echo "HuggingFace upload enabled: $HF_REPO"
fi

# NOTE: for C1-C4 byte targets, --max-tgt-len is auto-set by train.py to the
# longest plaintext line + 2 (the MAX_TGT positional arg is ignored for them).
python src/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --dim "$DIM" \
    --heads "$HEADS" \
    --kv-heads "$KV_HEADS" \
    --layers "$LAYERS" \
    --max-src-len "$MAX_SRC" \
    --max-tgt-len "$MAX_TGT" \
    --vocab-size "$VOCAB_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --byte-dim "$BYTE_DIM" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --output-dir outputs \
    $HF_ARGS \
    --wandb

EXIT_CODE=$?
echo "=========================================="
echo "Job finished $(date) with exit code $EXIT_CODE"
echo "Results: $DIR/outputs/$CONFIG/results.json"
echo "=========================================="
exit $EXIT_CODE
