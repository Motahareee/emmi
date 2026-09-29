#!/bin/bash
# Tests whether MobileCLIP QAT's Finding 23 gap (43.45% vs CLIP's
# 74.15%) is partly a training-budget artifact, the same way
# pruning-recovery's ~50-64% ceiling turned out to be (Finding 26:
# training actively overfits past a mid-training peak). Same 4000-image/
# 15-epoch budget as the original MobileCLIP QAT run -- only new variable
# is --early-stop, to isolate that one lever cleanly before also trying
# more data (Finding 23's root cause was "narrow trainable subset
# overfits," not demonstrated data-starvation, so this is the right
# first test).
#
# Feeds the early-stopped checkpoint into the SAME real-kernel
# quantization pipeline as train_qat_realquant.sh, so this also reports
# real size/latency, not just fake-quant accuracy.
#
# Run slurm/predownload_compression.sh on the login node FIRST.
#
# Usage: sbatch slurm/train_qat_realquant_mobileclip_earlystop.sh
#SBATCH --job-name=emma_qat_mc_earlystop
#SBATCH --output=logs/qat_mc_earlystop_%j.out
#SBATCH --error=logs/qat_mc_earlystop_%j.err
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

python3 -u train_qat_realquant.py \
    --n-train 4000 \
    --n-eval 1000 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --epochs 15 \
    --lr 1e-4 \
    --n-calibration 32 \
    --eval-every 3 \
    --early-stop \
    --models mobileclip \
    "$@"

echo "Done: $(date)"
