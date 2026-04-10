#!/bin/bash

#SBATCH --job-name=itk-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Used for preprocessing raw ITk samples into binary parquet files used for training

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
IN_DIR="/share/rcifdata/maxhart/data/itk/ATLAS-P2-RUN4-03-00-00_Rel.24_ttbar_uncorr_pu200_v9_acorn_data_reading_output_trainset/"
OUT_DIR="/share/rcifdata/maxhart/data/itk/train/"

# Python command that will be run
PYTORCH_CMD="python src/hepattn/experiments/itk/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR"

# Run via pixi
PIXI_CMD="pixi run $PYTORCH_CMD"
echo "Running: $PIXI_CMD"
$PIXI_CMD
echo "Done!"
