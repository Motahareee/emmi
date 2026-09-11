#!/bin/bash
# MC + VAE-64 sweep: mean/concat/match × 3 seeds (9 jobs)
# Plain β-VAE: MSE reconstruction + KL divergence loss.
# Usage: bash slurm/submit_mc_vae.sh

cd "$(dirname "$0")/.."

for FUS in mean concat match; do
    for SEED in 0 1 2; do
        TAG="emma_mc_${FUS}_vae_64_s${SEED}"
        sbatch -J "$TAG" slurm/train_encfus_80k.sh \
            vae 64 "$SEED" mobileclip "$FUS"
        echo "Submitted: $TAG"
    done
done

echo "Done — 9 jobs submitted."
