#!/bin/bash
# Submit all 5 ablation configurations (C1 to C5) in parallel to SLURM

set -e

# Location-independent: prioritize SLURM_SUBMIT_DIR, then cluster path, then script parent
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    DIR="$SLURM_SUBMIT_DIR"
elif [ -d "/home2/ojas.k/ANLP_a1/src" ]; then
    DIR="/home2/ojas.k/ANLP_a1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
cd "$DIR" || exit 1

mkdir -p outputs

EPOCHS=${1:-40}

echo "Submitting all 5 configs in parallel (EPOCHS=$EPOCHS)..."
for CFG in C1 C2 C3 C4 C5; do
    JOB_ID=$(sbatch --job-name="anlp_${CFG}" scripts/run_experiment.sh "$CFG" "$EPOCHS" | awk '{print $NF}')
    echo "  [Submitted] Config: $CFG -> SLURM Job ID: $JOB_ID"
done

echo "All 5 jobs submitted! Check status with: squeue -u $USER"
squeue -u "$USER" || true
