#!/bin/bash
# Diagnostic run: --debug-forward prints min/max/isnan/isinf at each stage of
# ServerPipeline.forward() (z -> projection -> combined -> each LLM layer's
# hidden_states -> final) for the first 3 forward calls, to pinpoint exactly
# where a non-finite value first appears for mobileclip+pca. Two prior fixes
# (grad-norm-skip guard, then input standardization) both failed to resolve
# it -- this measures the actual failure instead of guessing a third one.
#
# Capped to --max-epochs 2: we only need the first few batches' debug
# output, not a full training run.
#
# Usage: sbatch slurm/test_llava_pca_debug.sh
#SBATCH --job-name=emma_llava_mobileclip_pca_debug
#SBATCH --output=logs/llava_train_%j.out
#SBATCH --error=logs/llava_train_%j.err
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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
    --max-epochs 2 \
    --load-in-8bit \
    --debug-forward \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
