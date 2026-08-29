#!/bin/bash
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/%x_%j.log

# Location-independent: prioritize SLURM_SUBMIT_DIR, then cluster path, then script parent
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    DIR="$SLURM_SUBMIT_DIR"
elif [ -d "/home2/ojas.k/ANLP_a1/src" ]; then
    DIR="/home2/ojas.k/ANLP_a1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
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
PATCHING=${15:-entropy}
MAX_PATCH=${16:-12}
TARGET_PATCH_SIZE=${17:-4.0}
THETA_R=${18:-1.0}

# optional env overrides
RUN_NAME=${RUN_NAME:-$CONFIG}
OUT_DIR=${OUT_DIR:-outputs}
# set SKIP_HF=1 to skip the (tokenless) auto HF upload attempt
HF_ARGS=""
if [ -n "$HF_REPO" ]; then
    HF_ARGS="--hf-repo $HF_REPO"
elif [ "${SKIP_HF:-0}" != "1" ]; then
    HF_ARGS="--hf-repo IrishBumfuzzle/anlp-a1-$CONFIG"
fi

if [ "$CONFIG" = "C5" ] && [ -z "$3" ]; then
    BATCH_SIZE=16
    GRAD_ACCUM=1
fi

mkdir -p outputs
mkdir -p /scratch/$USER/tmp 2>/dev/null || true
export TMPDIR=/scratch/$USER/tmp

# Virtual env: prefer .venv_cluster (SLURM nodes), fall back to .venv
if [ -f "$DIR/.venv_cluster/bin/python" ]; then
    PY="$DIR/.venv_cluster/bin/python"
    source "$DIR/.venv_cluster/bin/activate" 2>/dev/null || true
elif [ -f "$DIR/.venv/bin/python" ]; then
    PY="$DIR/.venv/bin/python"
    source "$DIR/.venv/bin/activate" 2>/dev/null || true
elif [ -f "/home2/ojas.k/ANLP_a1/.venv_cluster/bin/python" ]; then
    PY="/home2/ojas.k/ANLP_a1/.venv_cluster/bin/python"
    source "/home2/ojas.k/ANLP_a1/.venv_cluster/bin/activate" 2>/dev/null || true
elif command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "ERROR: no python env found in $DIR"; exit 1
fi

echo "=========================================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $HOSTNAME"
echo "Repo:     $DIR"
echo "Start:    $(date)"
echo "Python:   $PY ($($PY --version 2>&1))"
echo "Config:   $CONFIG | Epochs: $EPOCHS | Batch: $BATCH_SIZE (x$GRAD_ACCUM accum) | LR: $LR | Dim: $DIM | Heads: $HEADS (kv: $KV_HEADS) | Layers: $LAYERS | MaxSrc: $MAX_SRC | MaxTgt: $MAX_TGT | Vocab: $VOCAB_SIZE"
echo "Patching: $PATCHING | max_patch: $MAX_PATCH | target mean: $TARGET_PATCH_SIZE | theta_r: $THETA_R | run: $RUN_NAME | out: $OUT_DIR"
echo "=========================================="

nvidia-smi || true

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-irishbumfuzzle-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-anlp-assignment1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -n "$HF_ARGS" ] && echo "HuggingFace upload enabled: $HF_ARGS"

$PY -u src/train.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
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
    --patching "$PATCHING" \
    --max-patch "$MAX_PATCH" \
    --target-patch-size "$TARGET_PATCH_SIZE" \
    --theta-r "$THETA_R" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --output-dir "$OUT_DIR" \
    $HF_ARGS \
    --wandb

EXIT_CODE=$?
echo "=========================================="
echo "Job finished $(date) with exit code $EXIT_CODE"
echo "Results: $DIR/$OUT_DIR/$CONFIG/results.json"
echo "=========================================="
exit $EXIT_CODE
