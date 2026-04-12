#!/bin/bash

#SBATCH --job-name=tide-pix-qcl
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
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
unset APPTAINER_BIND

PYTORCH_CMD="python src/reconstruct_anything/experiments/tide/main.py fit \
  -c src/reconstruct_anything/experiments/tide/configs/pix_only_qcl.yaml \
  -c src/reconstruct_anything/experiments/tide/configs/max.yaml"

PIXI_CMD="pixi run $PYTORCH_CMD"
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/pixi.sif $PIXI_CMD"

echo "Running: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
