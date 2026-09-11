#!/bin/bash
# Single-job validation of --standardize-input (train_llava.py's
# standardize_cache()): standardizes the projection's input to zero mean,
# unit variance (train-split stats) before it reaches LLaVA's fp16 backbone.
#
# mobileclip+pca is the known-worst case: with the NaN-guard alone (no
# standardization) it went non-finite on literally every batch and never
# learned anything (~50% accuracy, i.e. still random init, for all 5
# epochs). This is the fastest signal for whether standardization actually
# fixes the root cause (unbounded PCA output scale overflowing fp16) rather
# than just suppressing the symptom.
#
# If this trains normally (real accuracy movement, not stuck at ~50%),
# --standardize-input is validated and we can decide whether to apply it to
# the rest of the sweep (including re-running the 2 already-completed
# configs for consistency).
#
# Usage: sbatch slurm/test_llava_pca_standardize.sh
#SBATCH --job-name=emma_llava_mobileclip_pca_std
#SBATCH --output=logs/llava_train_%j.out
#SBATCH --error=logs/llava_train_%j.err
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
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
    --compression pca \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test 5000 \
    --batch-size 64 \
    --epochs 30 \
    --patience 5 \
    --load-in-8bit \
    --standardize-input \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
