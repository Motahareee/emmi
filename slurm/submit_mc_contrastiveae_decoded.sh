#!/bin/bash
# MC + ContrastiveAE-64 WITH server-side decoding: match × 3 seeds (3 jobs)
# Tests: does decoding back to 2048-dim before GPT-2 improve over raw 64-dim latents?
# Existing non-decoded results are preserved (separate cache dir).
# Usage: bash slurm/submit_mc_contrastiveae_decoded.sh

cd "$(dirname "$0")/.."

for SEED in 0 1 2; do
    TAG="emma_mc_match_contrastiveae_decoded_s${SEED}"
    sbatch -J "$TAG" slurm/train_encfus_80k.sh \
        contrastiveae 64 "$SEED" mobileclip match --decode-on-server
    echo "Submitted: $TAG"
done

echo "Done — 3 jobs submitted."
