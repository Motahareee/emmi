#!/bin/bash
# MC + CrossModalAE-64 sweep: mean/concat/match × 3 seeds (9 jobs)
# CrossModalAE: MSE + InfoNCE(z, text_emb, img_emb) — no binary labels needed.
# Uses the natural image-text pairing as free supervision.
# Usage: bash slurm/submit_mc_crossmodalae.sh

cd "$(dirname "$0")/.."

for FUS in mean concat match; do
    for SEED in 0 1 2; do
        TAG="emma_mc_${FUS}_crossmodalae_64_s${SEED}"
        sbatch -J "$TAG" slurm/train_encfus_80k.sh \
            crossmodalae 64 "$SEED" mobileclip "$FUS"
        echo "Submitted: $TAG"
    done
done

echo "Done — 9 jobs submitted."
