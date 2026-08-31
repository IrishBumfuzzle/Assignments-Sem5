#!/bin/bash
# Run all 5 configurations sequentially on ONE standalone GPU machine
# (target: RTX 2080 Ti, 11 GB, compute capability 7.5).
#
#   * C1-C4: byte-level targets (T ~ 2672), batch 8 x grad-accum 2 = 16
#            (the assignment's batch size). fp16 + GradScaler is selected
#            automatically on cc < 8.0 (train.py:get_amp_settings).
#   * C5 (BLT): non-autoregressive 1:1 byte mapping, batch 16 x 1 = 16.
#
# Requires: python env (.venv_cluster or .venv) from scripts/setup_cluster.sh,
#           data/brown_cipher.txt and data/brown_plain.txt,
#           a wandb API key (exported below by default), and optionally
#           HF_TOKEN (export it to upload checkpoints to HuggingFace when
#           each run finishes; without it the runs still complete, uploads
#           are skipped with a warning).
#
# Usage:   bash scripts/run_all.sh            # C1..C5, 40 epochs each
#          EPOCHS=10 bash scripts/run_all.sh  # shorter runs
#
# Wall time on a 2080 Ti: roughly 2.5-3 h per C1-C4 config, ~1.5 h for C5.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR" || exit 1

# --- python env -------------------------------------------------------------
if [ -f ".venv_cluster/bin/python" ]; then
    PY=".venv_cluster/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    echo "ERROR: no python env found. Run: bash scripts/setup_cluster.sh"; exit 1
fi
echo "Using python: $PY ($($PY --version 2>&1))"

# --- environment -------------------------------------------------------------
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-irishbumfuzzle-team}"
export WANDB_PROJECT="${WANDB_PROJECT:-anlp-assignment1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=${EPOCHS:-40}

nvidia-smi || true
mkdir -p outputs

echo "=========================================="
echo "OFFICIAL RUN START $(date)"
echo "Config: C1-C4 batch 8 x2, C5 batch 16 x1, epochs $EPOCHS"
echo "=========================================="

for CFG in C1 C2 C3 C4; do
    $PY -u src/train.py \
        --config "$CFG" \
        --epochs "$EPOCHS" \
        --batch-size 8 --grad-accum-steps 2 \
        --eval-batch-size 8 --num-workers 2 \
        --eval-greedy-every 4 \
        --run-name "$CFG" \
        --wandb --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
        --hf-repo "IrishBumfuzzle/anlp-a1-$CFG" \
        --output-dir "outputs/$CFG" 2>&1
    echo "=== $CFG DONE $(date) ==="
done

$PY -u src/train.py \
    --config C5 \
    --epochs "$EPOCHS" \
    --batch-size 16 --grad-accum-steps 1 \
    --eval-batch-size 16 --num-workers 2 \
    --eval-greedy-every 4 \
    --run-name C5 \
    --wandb --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" \
    --hf-repo "IrishBumfuzzle/anlp-a1-C5" \
    --output-dir "outputs/C5" 2>&1
echo "=== C5 DONE $(date) ==="
echo "ALL RUNS DONE $(date)"
echo
echo "Next steps:"
echo "  $PY scripts/make_results_table.py outputs   # LaTeX table for the report"
echo "  bash scripts/make_submission.sh <roll number>"
