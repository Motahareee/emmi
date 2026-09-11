#!/bin/bash
# SLURM worker for test-set evaluation.
# Evaluates one batch of checkpoints (3 at a time).
# Usage: sbatch -J eval_b0 slurm/eval_batch.sh 0
#        sbatch -J eval_b1 slurm/eval_batch.sh 1
#        (or use submit_eval.sh to fire all batches at once)
#
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

BATCH_IDX=${1:?usage: eval_batch.sh <batch_index>}

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "Batch idx: $BATCH_IDX"

python3 eval_batch.py --batch "$BATCH_IDX"

echo "Done: $(date)"
