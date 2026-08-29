#!/bin/bash
# Sync local code and scripts to ada cluster safely without overwriting venv, outputs, or wandb runs.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Syncing code to ojas.k@ada:/home2/ojas.k/ANLP_a1/ ..."
rsync -avz --delete \
    --exclude='.venv*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='outputs/' \
    --exclude='wandb/' \
    --exclude='*.zip' \
    ./ ojas.k@ada:/home2/ojas.k/ANLP_a1/

echo "Sync complete!"
