#!/bin/bash

#SBATCH --job-name=tide-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/slurm-%j.%x.out
#SBATCH --array 0-8

# Used for preprocessing raw tide samples into binary parquet files used for training

# Move to repo root relative to this script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
cd "$REPO_ROOT"
echo "Working directory: ${PWD}"

# Set tmpdir
export TMPDIR=/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
#IN_DIR="/share/rcifdata/maxhart/data/tide/raw/val/"
IN_DIR="/share/lustre/maxhart/data/ambi/user.srettie.800030.flatpT_Zprime_Extended.e7954_s3582_r12643_20241122_nom_with_rois_EXT1/"
OUT_DIR="/share/rcifdata/maxhart/data/ambi/tmp"

# Python command that will be run
# Note we specify a minimum pT cut, particles below this will be removed
PYTORCH_CMD="python src/reconstruct_anything/experiments/tide/prep.py -i $IN_DIR -o $OUT_DIR --parallel"

# Run via pixi
PIXI_CMD="pixi run $PYTORCH_CMD"
echo "Running: $PIXI_CMD"
$PIXI_CMD
echo "Done!"
