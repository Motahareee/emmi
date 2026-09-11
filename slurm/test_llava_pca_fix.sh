#!/bin/bash
# Single-job validation of the NaN-guard fix in train_llava.py's run_server_epoch
# (skips the optimizer step on a non-finite gradient instead of letting it
# permanently corrupt the model). mobileclip+pca is the fastest signal: in the
# pre-fix sweep it collapsed to NaN at epoch 1, so this tells us within the
# first epoch or two whether the guard actually works.
#
# If this comes back clean (or shows "[warn] non-finite grad norm ...
# -- skipping optimizer step" without the run collapsing), resubmit the full
# 13-job sweep with slurm/submit_llava_sweep.sh.
#
# Usage: sbatch slurm/test_llava_pca_fix.sh
#SBATCH --job-name=emma_llava_mobileclip_pca
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
    --checkpoint-dir checkpoints

echo "Done: $(date)"
