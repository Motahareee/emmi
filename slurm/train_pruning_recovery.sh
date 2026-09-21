#!/bin/bash
# Structured pruning + recovery fine-tuning for CLIP, at real scale --
# same fix as train_qat_compression.sh, for the same reason (report
# Finding 14): recovery fine-tuned on only 128 COCO photos made held-out
# cos_sim better (0.519 -> 0.646) but real zero-shot accuracy WORSE
# (21.0% -> 16.0%), independently confirming QAT's domain-mismatch
# finding on a completely different compression technique.
#
# Run slurm/predownload_compression.sh on the login node FIRST.
#
# Usage: sbatch slurm/train_pruning_recovery.sh
#   sbatch slurm/train_pruning_recovery.sh --ratio 0.2   # sweep other ratios
#SBATCH --job-name=emma_pruning_recovery
#SBATCH --output=logs/pruning_recovery_%j.out
#SBATCH --error=logs/pruning_recovery_%j.err
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
    --n-train 4000 \
    --n-eval 1000 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --epochs 15 \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
