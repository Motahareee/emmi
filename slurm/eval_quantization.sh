#!/bin/bash
# Quantization baseline evaluation — tests INT16/8/4/2/1 on the full
# 2048-dim match-fused embedding using the existing none server checkpoint.
# No retraining required.
#
# Usage:
#   sbatch slurm/eval_quantization.sh
#   sbatch slurm/eval_quantization.sh mobileclip match
#
#SBATCH --job-name=emma_quant
#SBATCH --output=logs/quant_%j.out
#SBATCH --error=logs/quant_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

ENCODER=${1:-mobileclip}
FUSION=${2:-match}

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "Encoder: $ENCODER"
echo "Fusion:  $FUSION"
echo ""

python3 eval_quantization.py \
    --encoder "$ENCODER" \
    --fusion  "$FUSION"  \
    --checkpoint-dir checkpoints \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test  5000

echo ""
echo "Done: $(date)"
