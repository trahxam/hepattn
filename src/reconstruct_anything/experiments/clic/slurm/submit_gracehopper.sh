#!/bin/bash

#SBATCH --job-name=clic-train
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Print host info
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Training command
CONFIG="src/reconstruct_anything/experiments/clic/configs/base.yaml"
PYTORCH_CMD="python src/reconstruct_anything/experiments/clic/main.py fit --config $CONFIG"

# Run via pixi (srun for GraceHopper multi-GPU)
PIXI_CMD="srun pixi run -e gracehopper $PYTORCH_CMD"
echo "Running: $PIXI_CMD"
$PIXI_CMD
echo "Done!"
