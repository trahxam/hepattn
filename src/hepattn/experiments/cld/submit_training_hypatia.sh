#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --output=/home/syw24/ftag/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out


# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /home/syw24/ftag/hepattn
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/home/syw24/tmp/

# Run the training
echo "Running training script..."

# Python command that will be run
# PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config src/hepattn/experiments/cld/configs/base.yaml "
# PYTORCH_CMD="python src/hepattn/experiments/cld/main.py test --config logs/CLD_10_96_TF_charged_10MeV_10GA_20250526-T225156/config.yaml --ckpt_path logs/CLD_10_96_TF_charged_10MeV_10GA_20250526-T225156/ckpts/epoch=009-train_loss=20.88666.ckpt "
PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config logs/CLD_10_96_TF_charged_10MeV_10GA_20250526-T225156/config.yaml --ckpt_path logs/CLD_10_96_TF_charged_10MeV_10GA_20250526-T225156/ckpts/epoch=009-train_loss=20.88666_train_eval.h5 "

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# APPTAINER_CMD="apptainer run --nv --bind /home/syw24 /home/syw24/ftag/hepattn/pixi.sif $PIXI_CMD"
APPTAINER_CMD="apptainer run --nv --bind /home/syw24 --bind /share/rcifdata/maxhart /home/syw24/ftag/hepattn/pixi.sif $PIXI_CMD"
  
# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
