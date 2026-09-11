#!/bin/bash
# Resubmit the 22 configs that hit TIMEOUT in the comprehensive run
# (all started simultaneously and stalled on raw-pickle I/O contention).
#
# train.py now skips raw data loading when the embed cache exists, so these
# go straight to compression/server training. Wall time raised to 8h anyway.
#
# Missing: none {s3,s4}, pca_32 {s0-s4}, pca_64 {s0,s1}, pca_128 {s0-s4},
#          ae_32 {s2,s3,s4}, ae_64 {s0-s4}   = 22 jobs

cd "$(dirname "$0")/.."

if [ ! -f checkpoints/embed_cache_72000_5000_5000/train_fused.pt ]; then
    echo "ERROR: shared embed cache not found — aborting."
    exit 1
fi

T="--time=08:00:00"

for SEED in 3 4; do
    sbatch $T -J "emma_none_s${SEED}" slurm/train_80k.sh none 0 "$SEED"
done

for SEED in 0 1 2 3 4; do
    sbatch $T -J "emma_pca_32_s${SEED}"  slurm/train_80k.sh pca 32  "$SEED"
    sbatch $T -J "emma_pca_128_s${SEED}" slurm/train_80k.sh pca 128 "$SEED"
    sbatch $T -J "emma_ae_64_s${SEED}"   slurm/train_80k.sh ae  64  "$SEED"
done

for SEED in 0 1; do
    sbatch $T -J "emma_pca_64_s${SEED}"  slurm/train_80k.sh pca 64 "$SEED"
done

for SEED in 2 3 4; do
    sbatch $T -J "emma_ae_32_s${SEED}"   slurm/train_80k.sh ae 32 "$SEED"
done

echo "Submitted 22 rerun jobs."
