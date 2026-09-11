#!/bin/bash
#SBATCH --job-name=emma_pca_80k
#SBATCH --output=logs/pca_80k_%j.out
#SBATCH --error=logs/pca_80k_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
# HF_DATASETS_OFFLINE disabled: valid/test caches need to be streamed once for 80k offsets

cd "$SLURM_SUBMIT_DIR"

echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Train.py:   $(grep -c 'n-train' train.py) occurrences of n-train in train.py"
echo "Python:     $(which python3)"

python3 train.py \
    --compression pca \
    --latent-dim 64 \
    --n-train 80000 \
    --batch-size 64 \
    --epochs 30 \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
