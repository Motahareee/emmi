#!/bin/bash
# MC + AE-64 and VAE-64 sweep: mean/concat/match × 3 seeds
# Both with and without server-side decoding.
# Total: 3 fusions × 3 seeds × 2 methods × 2 decode modes = 36 jobs
# Usage: bash slurm/submit_mc_ae_vae.sh

cd "$(dirname "$0")/.."

for METHOD in ae vae; do
    for FUS in mean concat match; do
        for SEED in 0 1 2; do
            # Without decoding (server sees raw 64-dim latent)
            TAG="emma_mc_${FUS}_${METHOD}_64_s${SEED}"
            sbatch -J "$TAG" slurm/train_encfus_80k.sh \
                "$METHOD" 64 "$SEED" mobileclip "$FUS"
            echo "Submitted: $TAG"

            # With server-side decoding (server sees reconstructed full-dim embedding)
            TAG="emma_mc_${FUS}_${METHOD}_decoded_64_s${SEED}"
            sbatch -J "$TAG" slurm/train_encfus_80k.sh \
                "$METHOD" 64 "$SEED" mobileclip "$FUS" --decode-on-server
            echo "Submitted: $TAG"
        done
    done
done

echo "Done — 36 jobs submitted."
