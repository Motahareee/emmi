#!/bin/bash
# Generic 80K training job. Usage:
#   sbatch -J emma_<method>_<dim>_s<seed>_80k slurm/train_80k.sh <compression> <latent_dim> [seed]
# e.g.
#   sbatch -J emma_vae_64_s1_80k slurm/train_80k.sh vae 64 1
#SBATCH --output=logs/80k_%x_%j.out
#SBATCH --error=logs/80k_%x_%j.err
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

COMPRESSION=${1:?usage: train_80k.sh <compression> <latent_dim> [seed]}
LATENT_DIM=${2:-64}
SEED=${3:-0}

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Compression: $COMPRESSION"
echo "Latent dim:  $LATENT_DIM"
echo "Seed:        $SEED"

if [ "$COMPRESSION" = "none" ]; then
    python3 train.py \
        --compression none \
        --n-train 72000 \
        --n-valid 5000 \
        --n-test 5000 \
        --batch-size 64 \
        --epochs 30 \
        --seed "$SEED" \
        --checkpoint-dir checkpoints
else
    python3 train.py \
        --compression "$COMPRESSION" \
        --latent-dim "$LATENT_DIM" \
        --n-train 72000 \
        --n-valid 5000 \
        --n-test 5000 \
        --batch-size 64 \
        --comp-epochs 50 \
        --epochs 30 \
        --seed "$SEED" \
        --checkpoint-dir checkpoints
fi

echo "Done: $(date)"
