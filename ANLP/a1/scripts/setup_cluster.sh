#!/bin/bash
# One-time setup on a SLURM compute node (or any Linux machine with CUDA):
#   srun -p u22 -C 2080ti --gres=gpu:1 bash scripts/setup_cluster.sh
# Creates .venv_cluster with PyTorch (CUDA) + all project dependencies.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================="
echo "Setting up Python environment in $DIR"
echo "Hostname: $HOSTNAME"
echo "Date:     $(date)"
echo "=========================================="

# Find python binary (prefer python3 / python3.10 / python3.12)
PY_BIN=""
for py in "${1:-}" python3 python3.10 python3.12 python; do
    if [ -n "$py" ] && command -v "$py" >/dev/null 2>&1; then
        PY_BIN=$(command -v "$py")
        break
    fi
done

if [ -z "$PY_BIN" ]; then
    echo "ERROR: No suitable Python interpreter found."
    exit 1
fi

echo "Using Python: $PY_BIN ($($PY_BIN --version 2>&1))"

# Check if uv is available, else use standard venv + pip
if command -v uv >/dev/null 2>&1; then
    echo "Found uv, using fast uv installer..."
    uv venv .venv_cluster --python "$PY_BIN"
    uv pip install --python .venv_cluster/bin/python torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv_cluster/bin/python tokenizers numpy matplotlib wandb huggingface_hub tqdm requests
else
    echo "Using standard venv and pip..."
    $PY_BIN -m venv .venv_cluster
    .venv_cluster/bin/pip install --upgrade pip
    .venv_cluster/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
    .venv_cluster/bin/pip install tokenizers numpy matplotlib wandb huggingface_hub tqdm requests
fi

echo "=========================================="
echo "Testing environment and CUDA support..."
.venv_cluster/bin/python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:    ', torch.cuda.get_device_name(0))
"
echo "=========================================="
echo "Environment setup successfully at: $DIR/.venv_cluster"
echo "=========================================="
