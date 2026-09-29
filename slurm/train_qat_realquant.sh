#!/bin/bash
# Real-kernel quantization of QAT-trained weights -- closes the gap left
# by train_qat_compression.sh, whose 74.15%/43.45% results are accuracy
# only: QAT's fake-quant never itself produces a real (smaller, faster)
# model, and nothing from those runs was ever saved or exported to a
# real int8 kernel. This runs QAT training with the SAME hyperparameters
# as train_qat_compression.sh, then immediately real-quantizes the
# trained weights (dynamic PTQ + static PTQ via ONNX Runtime) and
# measures real size/latency/zero-shot accuracy at every stage, in one
# job -- directly answers "what do we actually get if we deploy this."
#
# Run slurm/predownload_compression.sh on the login node FIRST.
#
# Usage: sbatch slurm/train_qat_realquant.sh
#   sbatch slurm/train_qat_realquant.sh --models clip   # CLIP only, faster
#SBATCH --job-name=emma_qat_realquant
#SBATCH --output=logs/qat_realquant_%j.out
#SBATCH --error=logs/qat_realquant_%j.err
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
    "$@"

echo "Done: $(date)"
