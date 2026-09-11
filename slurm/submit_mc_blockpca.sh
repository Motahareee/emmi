#!/bin/bash
# MC + BlockPCA-64 sweep: match fusion only × 3 seeds (3 jobs)
# BlockPCA applies PCA per block of [t; v; |t-v|; t⊙v].
# Only valid with match fusion (the 4 blocks are the input).
# Usage: bash slurm/submit_mc_blockpca.sh

cd "$(dirname "$0")/.."

for SEED in 0 1 2; do
    TAG="emma_mc_match_blockpca_64_s${SEED}"
    sbatch -J "$TAG" slurm/train_encfus_80k.sh \
        blockpca 64 "$SEED" mobileclip match
    echo "Submitted: $TAG"
done

echo "Done — 3 jobs submitted."
