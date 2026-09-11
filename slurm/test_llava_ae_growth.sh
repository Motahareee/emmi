#!/bin/bash
# Verification run before committing to any fix: does the projection's
# weight norm / soft-token output magnitude actually grow over the course
# of training, correlating with when non-finite grads start appearing? Or
# is the "runaway activation growth" theory wrong too, like the previous
# two hypotheses (grad-norm guard alone, input standardization)?
#
# mobileclip+ae chosen because it showed the largest early activations
# (last_hidden up to 67.69 vs ~20-30 for others) and is stuck at exactly
# 50% (fastest, clearest failure signal) -- if growth is going to show up
# anywhere, it should show up here.
#
# --debug-every 20 gives a snapshot roughly every 20 batches across the
# whole first epoch (~2250 train batches -> ~112 snapshot lines), capped
# at --max-epochs 1 since we only need to see the trend, not a full run.
#
# Usage: sbatch slurm/test_llava_ae_growth.sh
#SBATCH --job-name=emma_llava_mobileclip_ae_growth
#SBATCH --output=logs/llava_train_%j.out
#SBATCH --error=logs/llava_train_%j.err
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"

python3 -u train_llava.py \
    --encoder mobileclip \
    --fusion match \
    --compression ae \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test 5000 \
    --batch-size 64 \
    --max-epochs 1 \
    --load-in-8bit \
    --debug-forward \
    --debug-every 20 \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
