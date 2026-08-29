#!/bin/bash
#SBATCH --job-name=anlp_all
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/%x_%A_%a.log
#SBATCH --array=1-5%5

# Run all 5 configurations in PARALLEL on SLURM or locally across GPUs.
#
# Configurations:
#   Task 1 -> C1 (Base: Sinusoidal PE, MHA, LayerNorm, Byte Targets)
#   Task 2 -> C2 (RoPE: Rotary PE, MHA, LayerNorm, Byte Targets)
#   Task 3 -> C3 (GQA: Sinusoidal PE, Grouped-Query Attn kv_heads=4, LayerNorm, Byte Targets)
#   Task 4 -> C4 (RMSNorm: Sinusoidal PE, MHA, RMSNorm, Byte Targets)
#   Task 5 -> C5 (BLT: Byte Latent Transformer, Token-Free Patching)
#
# Usage:
#   sbatch scripts/run_all.sh          # Submits 5 parallel tasks on SLURM (Job Array)
#   bash scripts/run_all.sh            # Submits 5 parallel SLURM jobs (or runs parallel locally)
#   EPOCHS=10 bash scripts/run_all.sh  # Shorter test runs

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
mkdir -p /scratch/$USER/tmp 2>/dev/null || true
export TMPDIR=/scratch/$USER/tmp

CONFIGS=("C1" "C2" "C3" "C4" "C5")

# --- Case 1: Running as a SLURM Job Array task ---
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    CFG="${CONFIGS[$TASK_IDX]}"
    echo "=========================================="
    echo "SLURM Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID -> Config: $CFG"
    echo "Node: $HOSTNAME | GPU: $CUDA_VISIBLE_DEVICES"
    echo "=========================================="
    exec bash scripts/run_experiment.sh "$CFG" "${EPOCHS:-40}"
fi

# --- Case 2: Running from interactive shell / head node with SLURM ---
if command -v sbatch &>/dev/null; then
    echo "=========================================="
    echo "SLURM detected. Submitting all 5 configs in parallel..."
    echo "=========================================="
    for CFG in "${CONFIGS[@]}"; do
        JOB_ID=$(sbatch --job-name="anlp_${CFG}" scripts/run_experiment.sh "$CFG" "${EPOCHS:-40}" | awk '{print $NF}')
        echo "  [Submitted] Config: $CFG -> SLURM Job ID: $JOB_ID"
    done
    echo "All 5 jobs submitted in parallel!"
    echo "Check queue status with: squeue -u $USER"
    squeue -u "$USER" || true
    exit 0
fi

# --- Case 3: Running locally without SLURM (background parallel processes) ---
echo "=========================================="
echo "No SLURM detected. Launching all 5 configurations in parallel locally..."
echo "=========================================="
PIDS=()
for CFG in "${CONFIGS[@]}"; do
    echo "Starting $CFG in background -> outputs/${CFG}.log ..."
    bash scripts/run_experiment.sh "$CFG" "${EPOCHS:-40}" > "outputs/${CFG}.log" 2>&1 &
    PIDS+=($!)
done

echo "Running background PIDs: ${PIDS[*]}"
echo "Waiting for all parallel jobs to complete..."

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=1
done

if [ "$FAILED" -eq 0 ]; then
    echo "All parallel runs completed successfully!"
    if [ -f "$DIR/.venv_cluster/bin/python" ]; then
        "$DIR/.venv_cluster/bin/python" scripts/make_results_table.py outputs || true
    elif [ -f "$DIR/.venv/bin/python" ]; then
        "$DIR/.venv/bin/python" scripts/make_results_table.py outputs || true
    fi
else
    echo "One or more runs encountered errors. Check outputs/*.log for details."
    exit 1
fi
