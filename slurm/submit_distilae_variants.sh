#!/bin/bash
# DistilAE improvement sweep — 64-dim, 72K/5K/5K splits, 3 seeds per variant.
#   5 variants x 3 seeds = 15 jobs
#
# Variants (baseline distilae λ=0.1, batch 64 is already covered by the
# comprehensive experiment — not repeated here):
#   ld0.5  : λ_distil = 0.5
#   ld1    : λ_distil = 1.0
#   do     : distillation loss only (no reconstruction MSE)
#   md     : diagonal masked out of similarity MSE
#   cb512  : compression trained with batch 512 (richer similarity matrices)
#
# PREREQUISITE: the shared embed cache must already exist —
#   checkpoints/embed_cache_72000_5000_5000/train_fused.pt
# (built by the first job of submit_comprehensive_80k.sh). Do NOT run this
# before that job has finished.
#
# If a winner emerges, run seeds 3 and 4 for that variant to get 5-seed stats.

cd "$(dirname "$0")/.."

if [ ! -f checkpoints/embed_cache_72000_5000_5000/train_fused.pt ]; then
    echo "ERROR: shared embed cache not found — wait for the comprehensive"
    echo "       experiment's first job to finish before submitting variants."
    exit 1
fi

for SEED in 0 1 2; do
    sbatch -J "emma_dv_ld0.5_64_s${SEED}" slurm/train_distilae_variant_80k.sh 64 "$SEED" --lambda-distil 0.5
    sbatch -J "emma_dv_ld1_64_s${SEED}"   slurm/train_distilae_variant_80k.sh 64 "$SEED" --lambda-distil 1.0
    sbatch -J "emma_dv_do_64_s${SEED}"    slurm/train_distilae_variant_80k.sh 64 "$SEED" --distil-only
    sbatch -J "emma_dv_md_64_s${SEED}"    slurm/train_distilae_variant_80k.sh 64 "$SEED" --distil-mask-diag
    sbatch -J "emma_dv_cb512_64_s${SEED}" slurm/train_distilae_variant_80k.sh 64 "$SEED" --comp-batch-size 512
done

echo "Submitted 15 DistilAE variant jobs."
