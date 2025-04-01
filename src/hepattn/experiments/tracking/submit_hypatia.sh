#!/bin/bash

#SBATCH --job-name=trackml
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/slurm_logs/slurm-%j.%x.out


# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# move to workdir
cd /share/rcifdata/maxhart/hepattn/
echo "Moved dir, now in: ${PWD}"

# set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# run the training
echo "Running training script..."

PYTORCH_CMD="python src/hepattn/experiments/trackml/main.py fit --config src/hepattn/experiments/trackml/configs/trackml_tracking.yaml"
PIXI_CMD="pixi run $PYTORCH_CMD"
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"
pixi run $PYTORCH_CMD
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done."




