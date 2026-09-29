#!/bin/bash
# Information-theoretic capacity test (report "Next steps" follow-up to
# Finding 25): four independent interventions -- more epochs (Finding 18),
# iterative scheduling (Finding 19), Taylor importance (Finding 22),
# relational recovery loss (Finding 25) -- ALL failed to move
# pruning-recovery's ~50-64% accuracy ceiling at ratio=0.3. Every one of
# those changed HOW recovery is done; none tested whether the ceiling is
# simply a training-budget artifact (undertrained) or intrinsic to how
# much capacity survives pruning at that ratio (no amount of training
# can recover information that's structurally gone).
#
# This scales BOTH data and epochs together (not traded off against each
# other, unlike the earlier epoch-tuning sweep at fixed n_train=4000):
# 12000 train images (3x) and 40 epochs (2.7x) -- roughly an order of
# magnitude more total training compute than the standard baseline runs.
# --eval-every 5 tracks the held-out cos_sim/zero-shot accuracy
# trajectory throughout training, not just the final number, so we can
# see whether it's still climbing at epoch 40 or plateaued much earlier.
#
# Run slurm/predownload_compression.sh on the login node FIRST (this
# needs the LARGER 13500-sample COCO cache it now also fetches).
#
# Usage: sbatch slurm/train_pruning_recovery_scale.sh
#SBATCH --job-name=emma_pruning_scale
#SBATCH --output=logs/pruning_scale_%j.out
#SBATCH --error=logs/pruning_scale_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-a40
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

python3 -u train_pruning_recovery.py \
    --ratio 0.3 \
    --n-train 12000 \
    --n-eval 1500 \
    --n-zeroshot 2000 \
    --batch-size 128 \
    --epochs 40 \
    --eval-every 5 \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
