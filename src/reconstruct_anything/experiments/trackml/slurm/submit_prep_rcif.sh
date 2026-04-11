#!/bin/bash

#SBATCH --job-name=trackml-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Used for preprocessing raw TrackML samples into binary parquet files used for training

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
IN_DIR="/share/rcifdata/maxhart/data/trackml/raw/train/"
OUT_DIR="/share/rcifdata/maxhart/data/trackml/prepped/train/"

# Python command that will be run
PYTORCH_CMD="python src/reconstruct_anything/experiments/trackml/prep/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR"

# Run via pixi
PIXI_CMD="pixi run $PYTORCH_CMD"
echo "Running: $PIXI_CMD"
$PIXI_CMD
echo "Done!"
