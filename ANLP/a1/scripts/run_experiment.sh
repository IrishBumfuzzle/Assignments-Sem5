#!/bin/bash
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/ojas.k/ANLP_a1/outputs/%x_%j.log

CONFIG=${1:-C1}
EPOCHS=${2:-15}
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

echo "=========================================="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "Config: $CONFIG | Epochs: $EPOCHS | Batch Size: $BATCH_SIZE | LR: $LR | Dim: $DIM | Heads: $HEADS | Layers: $LAYERS | MaxSrc: $MAX_SRC | MaxTgt: $MAX_TGT | Vocab: $VOCAB_SIZE | GradAccum: $GRAD_ACCUM"
echo "=========================================="

nvidia-smi

cd /home2/ojas.k/ANLP_a1 || exit 1
mkdir -p outputs
mkdir -p /scratch/$USER/tmp
export TMPDIR=/scratch/$USER/tmp

source .venv_cluster/bin/activate

export WANDB_API_KEY="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"
export WANDB_ENTITY="irishbumfuzzle-team"
export WANDB_PROJECT="anlp-assignment1"
export WANDB_MODE="online"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:$PYTHONPATH

python src/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --dim "$DIM" \
    --heads "$HEADS" \
    --layers "$LAYERS" \
    --max-src-len "$MAX_SRC" \
    --max-tgt-len "$MAX_TGT" \
    --vocab-size "$VOCAB_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --byte-dim "$BYTE_DIM" \
    --wandb-project "anlp-assignment1" \
    --wandb-entity "irishbumfuzzle-team" \
    --output-dir outputs \
    --wandb

EXIT_CODE=$?
echo "=========================================="
echo "Job Finished at: $(date) with exit code $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
