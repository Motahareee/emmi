#!/bin/bash
# Encoder/fusion variant job on the 72K/5K/5K splits. Usage:
#   sbatch -J <name> slurm/train_encfus_80k.sh <compression> <latent_dim> <seed> <encoder> <fusion>
# e.g.
#   sbatch -J emma_mc_match_none_s0 slurm/train_encfus_80k.sh none 0 0 mobileclip match
#   sbatch -J emma_clip_match_distilae_64_s1 slurm/train_encfus_80k.sh distilae 64 1 clip match
#
# Each (encoder, fusion) combo has its own embed cache — the first job of a
# combo builds it (needs the raw pkl caches; slow), later jobs reuse it.
#
# PREREQUISITE (login node, once):
#   pip install open_clip_torch
#   python3 -c "import open_clip; open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')"
#SBATCH --output=logs/ef_%x_%j.out
#SBATCH --error=logs/ef_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

COMPRESSION=${1:?usage: train_encfus_80k.sh <compression> <latent_dim> <seed> <encoder> <fusion> [extra flags...]}
LATENT_DIM=${2:-64}
SEED=${3:-0}
ENCODER=${4:-clip}
FUSION=${5:-mean}
shift 5
EXTRA_FLAGS=("$@")

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:        $SLURMD_NODENAME"
echo "Start:       $(date)"
echo "Compression: $COMPRESSION"
echo "Latent dim:  $LATENT_DIM"
echo "Seed:        $SEED"
echo "Encoder:     $ENCODER"
echo "Fusion:      $FUSION"

COMP_FLAGS=()
if [ "$COMPRESSION" != "none" ]; then
    COMP_FLAGS=(--latent-dim "$LATENT_DIM" --comp-epochs 50)
fi

python3 train.py \
    --compression "$COMPRESSION" \
    --encoder "$ENCODER" \
    --fusion "$FUSION" \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test 5000 \
    --batch-size 64 \
    --epochs 30 \
    --seed "$SEED" \
    --checkpoint-dir checkpoints \
    "${COMP_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}"

echo "Done: $(date)"
