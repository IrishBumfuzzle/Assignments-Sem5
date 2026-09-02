#!/bin/bash
# Submit the two C5 SLURM jobs (entropy patching = official C5, fixed-4 = ablation).
#
#   Job 1: C5-entropy  -- entropy-based dynamic patching (BLT paper Sec 2.3), 120 epochs
#   Job 2: C5-fixed    -- fixed stride-4 patches, same architecture, 120 epochs
#
# Prereqs: repo synced to the server (code + data + .venv_cluster), incl. the new
#   src/entropy_patching.py and updated src/models/blt.py, src/train.py, src/dataset.py.
#   e.g.  rsync -av --exclude .venv --exclude .venv_cluster --exclude outputs \
#         --exclude wandb --exclude '*.pt' ./ <server>:/home2/ojas.k/ANLP_a1/
#
# Usage:  bash scripts/submit_c5.sh            # via SLURM (sbatch)
#         NO_SRUN=1 bash scripts/submit_c5.sh  # run sequentially in the foreground

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    DIR="$SLURM_SUBMIT_DIR"
elif [ -d "/home2/ojas.k/ANLP_a1/src" ]; then
    DIR="/home2/ojas.k/ANLP_a1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
cd "$DIR" || exit 1

EPOCHS=${EPOCHS:-120}
export SKIP_HF=1          # user uploads checkpoints to HF manually afterwards

echo "Submitting C5 runs (EPOCHS=$EPOCHS) from $DIR"

submit() {
    local name="$1"; shift
    if command -v sbatch &>/dev/null && [ "${NO_SRUN:-0}" != "1" ]; then
        local job
        job=$(sbatch --job-name="anlp_$name" "$@" | awk '{print $NF}')
        echo "  [Submitted] $name -> SLURM job $job"
    else
        echo "  [Foreground] $name"
        "$@"
    fi
}

# 1) official C5: entropy-based dynamic patching -> outputs/C5
RUN_NAME=C5-entropy OUT_DIR=outputs \
    submit C5_entropy bash scripts/run_experiment.sh C5 "$EPOCHS"

# 2) ablation: fixed stride-4 patches, same architecture -> outputs/ablations/C5
RUN_NAME=C5-fixed OUT_DIR=outputs/ablations \
    submit C5_fixed bash scripts/run_experiment.sh C5 "$EPOCHS" 16 5e-4 256 8 4 1024 512 4 64 8000 1 4 fixed

echo
echo "Done. Results:"
echo "  C5-entropy -> outputs/C5/results.json"
echo "  C5-fixed   -> outputs/ablations/C5/results.json"
