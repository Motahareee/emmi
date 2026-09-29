#!/bin/bash
# New canonical pruning-recovery result at ratio=0.3 (Finding 26 follow-up).
# The capacity test (train_pruning_recovery_scale.sh) proved the ~50-64%
# ceiling wasn't intrinsic -- accuracy peaked at 69.85% (epoch 25/40)
# before overfitting dragged it back down to 63.95% by the final epoch.
# This reruns the same 12000-image/40-epoch budget with --early-stop
# (checkpoints on best held-out zero-shot accuracy, restores that
# checkpoint instead of the final epoch) and --eval-every 2 for tighter
# resolution around the peak than the capacity test's --eval-every 5.
#
# This result replaces 49.65% as the canonical pruning-recovery number
# used in the CLIP-vs-MobileCLIP research-question comparison.
#
# Run slurm/predownload_compression.sh on the login node FIRST if you
# haven't already (needs the 13500-sample COCO cache).
#
# Usage: sbatch slurm/train_pruning_recovery_earlystop.sh
#SBATCH --job-name=emma_pruning_earlystop
#SBATCH --output=logs/pruning_earlystop_%j.out
#SBATCH --error=logs/pruning_earlystop_%j.err
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
    --eval-every 2 \
    --early-stop \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
