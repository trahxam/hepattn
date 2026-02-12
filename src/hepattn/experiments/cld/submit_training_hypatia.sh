#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out


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
cd /share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/share/rcifdata/maxhart/tmp/

# Run the training
echo "Running training script..."   

echo "CUDA visible devices is ${CUDA_VISIBLE_DEVICES}"

# Python command that will be run
#CONFIG_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_2_320_10MeV_neutrals_20251026-T230553/config.yaml"
#CKPT_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_2_320_10MeV_neutrals_20251026-T230553/ckpts/epoch=000-train_loss=3.05229.ckpt"
#PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config $CONFIG_PATH --ckpt_path $CKPT_PATH"

PYTORCH_CMD="python main.py fit --config configs/unified.yaml --config configs/kmax.yaml --name CLD_Combined_SmallBCE"

#PYTORCH_CMD="python /share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/main.py fit \
#-c /share/rcifdata/maxhart/hepattn/logs/CLD_5_256_10MeV_20251205-T083126/config.yaml \
#--ckpt_path /share/rcifdata/maxhart/hepattn/logs/CLD_5_256_10MeV_20251205-T083126/ckpts/epoch=015-val_loss=5.27151.ckpt"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart,/share/lustre/maxhart,/share/rcif2/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
