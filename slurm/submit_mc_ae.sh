#!/bin/bash
# MC + AE-64 sweep: mean/concat/match × 3 seeds (9 jobs)
# Baseline for ContrastiveAE ablation — shows what SupCon loss adds.
# Usage: bash slurm/submit_mc_ae.sh

cd "$(dirname "$0")/.."

for FUS in mean concat match; do
    for SEED in 0 1 2; do
        TAG="emma_mc_${FUS}_ae_64_s${SEED}"
        sbatch -J "$TAG" slurm/train_encfus_80k.sh \
            ae 64 "$SEED" mobileclip "$FUS"
        echo "Submitted: $TAG"
    done
done

echo "Done — 9 jobs submitted."
