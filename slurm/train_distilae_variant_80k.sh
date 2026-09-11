#!/bin/bash
# DistilAE variant job on the 72K/5K/5K splits. Usage:
#   sbatch -J emma_dv_<tag> slurm/train_distilae_variant_80k.sh <latent_dim> <seed> [extra train.py flags...]
# e.g.
#   sbatch -J emma_dv_md_64_s0 slurm/train_distilae_variant_80k.sh 64 0 --distil-mask-diag
#   sbatch -J emma_dv_ld1_64_s0 slurm/train_distilae_variant_80k.sh 64 0 --lambda-distil 1.0
#
# Reuses the shared embed cache built by the comprehensive experiment —
# only submit after checkpoints/embed_cache_72000_5000_5000/train_fused.pt exists.
#SBATCH --output=logs/dv_%x_%j.out
#SBATCH --error=logs/dv_%x_%j.err
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

LATENT_DIM=${1:?usage: train_distilae_variant_80k.sh <latent_dim> <seed> [flags...]}
SEED=${2:?usage: train_distilae_variant_80k.sh <latent_dim> <seed> [flags...]}
shift 2
EXTRA_FLAGS=("$@")

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:        $SLURMD_NODENAME"
echo "Start:       $(date)"
echo "Latent dim:  $LATENT_DIM"
echo "Seed:        $SEED"
echo "Variant:     ${EXTRA_FLAGS[*]}"

python3 train.py \
    --compression distilae \
    --latent-dim "$LATENT_DIM" \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test 5000 \
    --batch-size 64 \
    --comp-epochs 50 \
    --epochs 30 \
    --seed "$SEED" \
    --checkpoint-dir checkpoints \
    "${EXTRA_FLAGS[@]}"

echo "Done: $(date)"
