#!/bin/bash
# Relational/contrastive distillation loss (report Finding 22 follow-up)
# instead of plain feature cosine-similarity, at the SAME ratio=0.3 (same
# L2-magnitude pruning as the baseline) for a direct comparison against
# cosine-loss recovery's 49.65% accuracy / 0.866 cos_sim. Three
# independent negative results (more epochs, iterative scheduling,
# Taylor importance) all failed to move that ceiling while holding the
# recovery LOSS fixed -- this is the first experiment that changes the
# loss itself instead of how pruning/training is scheduled or scored.
#
# Run slurm/predownload_compression.sh on the login node FIRST if you
# haven't already (shares the same cached CLIP/CIFAR-10/COCO data).
#
# Usage: sbatch slurm/train_pruning_recovery_relational.sh
#   sbatch slurm/train_pruning_recovery_relational.sh --ratio 0.2   # other ratios
#SBATCH --job-name=emma_pruning_relational
#SBATCH --output=logs/pruning_relational_%j.out
#SBATCH --error=logs/pruning_relational_%j.err
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

python3 -u train_pruning_recovery.py \
    --ratio 0.3 \
    --recovery-loss relational \
    --n-train 4000 \
    --n-eval 1000 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --epochs 15 \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
