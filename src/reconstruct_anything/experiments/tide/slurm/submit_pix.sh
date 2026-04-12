#!/bin/bash

#SBATCH --job-name=tide-pix
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/reconstruct_anything/experiments/tide/slurm_logs/slurm-%j.%x.out
#SBATCH --error=/share/rcifdata/maxhart/hepattn/src/reconstruct_anything/experiments/tide/slurm_logs/slurm-%j.%x.out

echo "Setting comet experiment key"
timestamp=$(date +%s)
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /share/rcifdata/maxhart/hepattn/
echo "Working dir: ${PWD}"

export TMPDIR=/share/rcifdata/maxhart/tmp/
unset APPTAINER_BIND  # don't inherit login-node slurm binds into the job container

PYTORCH_CMD="python src/reconstruct_anything/experiments/tide/main.py fit \
  -c src/reconstruct_anything/experiments/tide/configs/base.yaml \
  -c src/reconstruct_anything/experiments/tide/configs/max.yaml"

PIXI_CMD="pixi run -e l40s $PYTORCH_CMD"
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/pixi.sif $PIXI_CMD"

echo "Running: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
