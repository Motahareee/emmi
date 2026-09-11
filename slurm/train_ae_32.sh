#!/bin/bash
#SBATCH --job-name=emma_ae_32
#SBATCH --output=logs/ae_32_%j.out
#SBATCH --error=logs/ae_32_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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

cd "$SLURM_SUBMIT_DIR"

echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"

python3 train.py \
    --compression ae \
    --latent-dim 32 \
    --batch-size 64 \
    --comp-epochs 50 \
    --epochs 30 \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
