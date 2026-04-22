#!/bin/bash
#SBATCH --job-name=eval-1k
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=compute-gpu-0-4
#SBATCH --output=/share/gpu1/syw24/hepattn-pr/src/hepattn/experiments/cld/scripts/slurm_logs/slurm-%j.%x.out

echo "Hostname: $(hostname)"
nvidia-smi | head -20

PR=/share/gpu1/syw24/hepattn-pr
CONFIG="$PR/src/hepattn/experiments/cld/eval_configs/config_eval_ttbar_fixed.yaml"
CKPT="/share/gpu1/syw24/hepattn/logs/CLD_kmax_frozen_property_regression_v7_ablation_20260326-T155114/ckpts/epoch=000-val_loss=15.70569.ckpt"

cd "$PR"
export TMPDIR=/share/gpu1/syw24/tmp/

CMD="pixi run python $PR/src/hepattn/experiments/cld/main.py test --config $CONFIG --ckpt_path $CKPT"
APPTAINER_CMD="apptainer run --nv --bind /share/gpu1/syw24 --bind /share/rcif2 --home /share/gpu1/syw24 $PR/pixi.sif bash -c \"$CMD\""

echo "Running:"
echo "$APPTAINER_CMD"
echo ""
eval "$APPTAINER_CMD"
RC=$?
echo "Exit code: $RC"
echo "Done!"
exit $RC
