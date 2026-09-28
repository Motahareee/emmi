#!/bin/bash
# Iterative pruning (prune a little, fine-tune, repeat) vs.
# train_pruning_recovery.py's one-shot approach at the same final ratio.
# One-shot ratio=0.3 + recovery got 49.65% real zero-shot accuracy
# (report Finding 17) -- this tests whether spreading the same total cut
# across 3 smaller steps preserves more accuracy, per the standard
# pruning-literature result that one-shot pruning damages more per unit
# of compression than iterative pruning does.
#
# Run slurm/predownload_compression.sh on the login node FIRST if you
# haven't already (shares the same cached CLIP/CIFAR-10/COCO data).
#
# Usage: sbatch slurm/train_iterative_pruning.sh
#   sbatch slurm/train_iterative_pruning.sh --target-ratio 0.1   # other ratios
#SBATCH --job-name=emma_iterative_pruning
#SBATCH --output=logs/iterative_pruning_%j.out
#SBATCH --error=logs/iterative_pruning_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
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

python3 -u train_iterative_pruning.py \
    --target-ratio 0.3 \
    --n-steps 3 \
    --epochs-per-step 5 \
    --final-epochs 15 \
    --n-train 4000 \
    --n-eval 1000 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
