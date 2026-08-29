#!/bin/bash
# Official C1-C4 retrain at BPE vocab 512 (40 epochs each, 4 parallel SLURM jobs).
#
# Why: the vocab-512 probe (15 ep) reached 0.950 test bit acc / 8.3% seq acc,
# vs 0.690 / 0.6% for the official 8000-vocab runs at 40 ep. Vocab 512 keeps
# the (byte,phase) source ~1:1 with byte positions, so greedy decoding stays
# aligned. 512 becomes the official tokenizer for C1-C4; the old 8000-vocab
# runs are kept as the tokenization ablation.
#
# Prereq ON THE SERVER (preserve the old 8000-vocab official outputs first):
#   cd /home2/ojas.k/ANLP_a1
#   mkdir -p outputs/ablations/vocab8000
#   mv outputs/C1 outputs/C2 outputs/C3 outputs/C4 outputs/ablations/vocab8000/
#   [ -d outputs/ablations/C1 ] && mv outputs/ablations/C1 outputs/ablations/probe-c1-v512-ep15
#
# Usage:  bash scripts/submit_c14_vocab512.sh            # via SLURM (sbatch)
#         NO_SRUN=1 bash scripts/submit_c14_vocab512.sh  # foreground (serial)

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src" ]; then
    DIR="$SLURM_SUBMIT_DIR"
elif [ -d "/home2/ojas.k/ANLP_a1/src" ]; then
    DIR="/home2/ojas.k/ANLP_a1"
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
fi
cd "$DIR" || exit 1

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-8}
export SKIP_HF=1          # user uploads checkpoints to HF manually afterwards

echo "Submitting official C1-C4 @vocab-512 (EPOCHS=$EPOCHS, BATCH=$BATCH) from $DIR"

for C in C1 C2 C3 C4; do
    if command -v sbatch &>/dev/null && [ "${NO_SRUN:-0}" != "1" ]; then
        job=$(RUN_NAME="${C}-v512" OUT_DIR=outputs SKIP_HF=1 \
              sbatch --job-name="anlp_${C}_v512" \
              --export=ALL,RUN_NAME="${C}-v512",OUT_DIR=outputs,SKIP_HF=1 \
              scripts/run_experiment.sh "$C" "$EPOCHS" "$BATCH" 5e-4 256 8 4 1024 512 4 64 512 2 4 \
              | awk '{print $NF}')
        echo "  [Submitted] $C-v512 -> SLURM job $job"
    else
        echo "  [Foreground] $C-v512"
        RUN_NAME="${C}-v512" OUT_DIR=outputs SKIP_HF=1 \
            bash scripts/run_experiment.sh "$C" "$EPOCHS" "$BATCH" 5e-4 256 8 4 1024 512 4 64 512 2 4
    fi
done

echo
echo "Done. Official results -> outputs/C{1..4}/results.json"
echo "Old 8000-vocab runs    -> outputs/ablations/vocab8000/C{1..4}/"
