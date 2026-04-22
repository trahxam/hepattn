#!/bin/bash
#SBATCH --job-name=regrtask-validate-plots
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/share/gpu1/syw24/hepattn-pr/src/hepattn/experiments/cld/scripts/slurm_logs/slurm-%j.%x.out

echo "Hostname: $(hostname)"
PR=/share/gpu1/syw24/hepattn-pr
cd "$PR"
export TMPDIR=/share/gpu1/syw24/tmp/

H5="/share/gpu1/syw24/hepattn/logs/CLD_kmax_frozen_property_regression_v7_ablation_20260326-T155114/ckpts/epoch=000-val_loss=15.70569_ttbar_fixed_eval.h5"
OUT="$PR/src/hepattn/experiments/cld/plots/baseline_v7_ablation_prtree_1k"

CMD="pixi run -e cpu python $PR/src/hepattn/experiments/cld/scripts/baseline_plots_v7_ablation.py --h5 $H5 --n_events 1000 --out_dir $OUT"
APPTAINER_CMD="apptainer run --nv --bind /share/gpu1/syw24 --bind /share/rcif2 --home /share/gpu1/syw24 $PR/pixi.sif bash -c \"$CMD\""

echo "Running:"
echo "$APPTAINER_CMD"
echo ""
eval "$APPTAINER_CMD"
RC=$?
echo "Exit code: $RC"
echo "Done!"
exit $RC
