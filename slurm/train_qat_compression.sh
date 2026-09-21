#!/bin/bash
# QAT fine-tuning for CLIP + MobileCLIP, at real scale -- the direct fix
# for what the CPU investigation found (report Finding 13): QAT fine-tuned
# on only 128 COCO photos collapsed real zero-shot CIFAR-10 accuracy to
# 34-37% despite cos_sim looking reasonable (0.82 for CLIP), a domain-
# mismatch failure. This run uses ~30x more, more diverse fine-tuning
# data and a much larger zero-shot eval set to see whether that actually
# fixes it, or whether QAT needs something beyond "more data" here.
#
# Run slurm/predownload_compression.sh on the login node FIRST -- this
# job runs fully offline (TRANSFORMERS_OFFLINE etc. below).
#
# Usage: sbatch slurm/train_qat_compression.sh
#SBATCH --job-name=emma_qat_compression
#SBATCH --output=logs/qat_compression_%j.out
#SBATCH --error=logs/qat_compression_%j.err
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

# n-train/n-eval: 4000/1000 real COCO image-caption pairs, vs. the CPU
# investigation's 128/64 -- the direct fix for the domain-mismatch
# collapse. n-zeroshot: 2000 CIFAR-10 test images (vs. ~200 on CPU) for a
# more statistically solid real-accuracy readout. batch-size raised from
# CPU's memory-constrained 8 to 64, appropriate for A40 VRAM.
python3 -u train_qat.py \
    --n-train 4000 \
    --n-eval 1000 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --epochs 15 \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
