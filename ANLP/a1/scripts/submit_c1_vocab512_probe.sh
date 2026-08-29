#!/bin/bash
# Probe: C1 with BPE vocab 512 (instead of 8000), 15 epochs, to compare
# against the official C1's epoch-15 metrics (vocab 8000):
#
#   C1 baseline @ ep15 (vocab 8000, outputs/anlp_C1_2684007.log):
#     train_loss 0.9934 | val_loss 1.0621 (tf) | val bit_acc 0.8918 | lev 199.9
#     greedy val @ ep10: bit_acc 0.6792
#
# Vocab 512 covers all 424 observed (byte,phase) base symbols + 88 merges
# (0 UNK); source sequences are ~2.4x longer and the test split keeps
# 468/500 lines at the 1024-token cap.
#
# Usage:  bash scripts/submit_c1_vocab512_probe.sh           # via SLURM
#         NO_SRUN=1 bash scripts/submit_c1_vocab512_probe.sh # foreground

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    DIR="$SLURM_SUBMIT_DIR"
elif [ -d "/home2/ojas.k/ANLP_a1/src" ]; then
    DIR="/home2/ojas.k/ANLP_a1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
cd "$DIR" || exit 1

EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-8}        # if the node OOMs, re-run with BATCH=4
export SKIP_HF=1

echo "Submitting C1 vocab-512 probe (EPOCHS=$EPOCHS, BATCH=$BATCH) from $DIR"

if command -v sbatch &>/dev/null && [ "${NO_SRUN:-0}" != "1" ]; then
    JOB_ID=$(RUN_NAME=C1-vocab512 OUT_DIR=outputs/ablations SKIP_HF="${SKIP_HF:-1}" \
        sbatch --job-name="anlp_C1_v512" --export=ALL,RUN_NAME=C1-vocab512,OUT_DIR=outputs/ablations,SKIP_HF="${SKIP_HF:-1}" \
        scripts/run_experiment.sh C1 "$EPOCHS" "$BATCH" 5e-4 256 8 4 1024 512 4 64 512 2 4 \
        | awk '{print $NF}')
    echo "  [Submitted] C1-vocab512 -> SLURM job $JOB_ID"
else
    RUN_NAME=C1-vocab512 OUT_DIR=outputs/ablations SKIP_HF="${SKIP_HF:-1}" \
        bash scripts/run_experiment.sh C1 "$EPOCHS" "$BATCH" 5e-4 256 8 4 1024 512 4 64 512 2 4
fi

echo
echo "Done. Results: outputs/ablations/C1/results.json  (wandb run: C1-vocab512)"
