#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
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
cd /share/rcifdata/maxhart/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/share/rcifdata/maxhart/tmp/

# get list of valid gpu indices from nvidia-smi, e.g. "0 1 2 3"
VALID_GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))

# current assignment from slurm
REQ_GPU_ID="$CUDA_VISIBLE_DEVICES"

is_valid=false
for gid in "${VALID_GPU_IDS[@]}"; do
    if [ "$gid" = "$REQ_GPU_ID" ]; then
        is_valid=true
        break
    fi
done

if [ "$is_valid" = true ]; then
    echo "CUDA_VISIBLE_DEVICES is valid: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES) is invalid. Reassigning."

    # pick gpu with lowest memory.used
    DEV=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | sort -t, -k2 -n \
        | head -n1 \
        | cut -d',' -f1 \
        | xargs)

    export CUDA_VISIBLE_DEVICES="$DEV"
    echo "Using GPU: $DEV (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

# Run the training
echo "Running training script..."

# Python command that will be run
#CONFIG_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_2_320_10MeV_neutrals_20251026-T230553/config.yaml"
#CKPT_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_2_320_10MeV_neutrals_20251026-T230553/ckpts/epoch=000-train_loss=3.05229.ckpt"
#PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config $CONFIG_PATH --ckpt_path $CKPT_PATH"
PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config src/hepattn/experiments/cld/configs/base.yaml --config src/hepattn/experiments/cld/configs/harmonics.yaml"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"