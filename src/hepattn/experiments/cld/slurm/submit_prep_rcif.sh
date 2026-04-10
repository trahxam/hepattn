#!/bin/bash

#SBATCH --job-name=cld-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/slurm-%j.%x.out
#SBATCH --array 0-4

# Used for preprocessing raw CLD samples into binary parquet files used for training

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
#IN_DIR="/share/rcifdata/maxhart/data/cld/raw/val/"
IN_DIR="/share/rcif2/maxhart/data/cld/test/raw/"
OUT_DIR="/share/rcif2/maxhart/data/cld/test/prepped/"

# Python command that will be run
# Note we specify a minimum pT cut, particles below this will be removed
PYTORCH_CMD="python src/hepattn/experiments/cld/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR --min_pt 10 --parallel"

# Run via pixi
PIXI_CMD="pixi run $PYTORCH_CMD"
echo "Running: $PIXI_CMD"
$PIXI_CMD
echo "Done!"
