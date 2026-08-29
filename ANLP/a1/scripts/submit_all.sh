#!/bin/bash
# Submit all 5 ablation configurations (C1 to C5) to SLURM

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

mkdir -p outputs

echo "Submitting C1 (Base)..."
sbatch --job-name=anlp_C1 scripts/run_experiment.sh C1

echo "Submitting C2 (RoPE)..."
sbatch --job-name=anlp_C2 scripts/run_experiment.sh C2

echo "Submitting C3 (GQA)..."
sbatch --job-name=anlp_C3 scripts/run_experiment.sh C3

echo "Submitting C4 (RMSNorm)..."
sbatch --job-name=anlp_C4 scripts/run_experiment.sh C4

echo "Submitting C5 (BLT)..."
sbatch --job-name=anlp_C5 scripts/run_experiment.sh C5

echo "All 5 jobs submitted! Check status with: squeue -u \$USER"
