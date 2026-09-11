#!/bin/bash
# Submit 5-seed runs of the key 64-dim configs at 80K.
# Seed 0 results already exist from the previous round — submitting seeds 1-4.
cd "$(dirname "$0")/.."

for SEED in 1 2 3 4; do
    sbatch -J "emma_none_s${SEED}_80k"     slurm/train_80k.sh none     0  "$SEED"
    sbatch -J "emma_pca_64_s${SEED}_80k"   slurm/train_80k.sh pca      64 "$SEED"
    sbatch -J "emma_ae_64_s${SEED}_80k"    slurm/train_80k.sh ae       64 "$SEED"
    sbatch -J "emma_vae_64_s${SEED}_80k"   slurm/train_80k.sh vae      64 "$SEED"
    sbatch -J "emma_distilae_64_s${SEED}_80k" slurm/train_80k.sh distilae 64 "$SEED"
done

echo "All seed jobs submitted (5 methods x 4 seeds = 20 jobs; seed 0 already done)."
