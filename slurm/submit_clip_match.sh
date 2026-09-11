#!/bin/bash
# CLIP + match fusion compression sweep — fills missing table cells.
# Methods: none, pca, ae, vae, lda, blockpca, crossmodalae, contrastiveae
# Encoder: clip, Fusion: match, Seed: 0 (single seed — comparison baseline)
#
# The clip+match embed cache may already exist from submit_encfus.sh (none+distilae).
# If it exists, all jobs reuse it. If not, the none job builds it first.
#
# Usage: bash slurm/submit_clip_match.sh

cd "$(dirname "$0")/.."

ENC=clip
FUS=match

# none — builds embed cache if missing; other jobs depend on it
FIRST=$(sbatch --parsable -J "emma_clip_match_none_s0" \
    slurm/train_encfus_80k.sh none 0 0 "$ENC" "$FUS")
echo "Submitted: emma_clip_match_none_s0 (job $FIRST)"
DEP="--dependency=afterok:$FIRST"

# Closed-form methods (fast, no GPU needed but submitted to same queue)
for METHOD in pca lda blockpca; do
    sbatch $DEP -J "emma_clip_match_${METHOD}_64_s0" \
        slurm/train_encfus_80k.sh "$METHOD" 64 0 "$ENC" "$FUS"
    echo "Submitted: emma_clip_match_${METHOD}_64_s0"
done

# Neural methods (50 compression epochs + 30 server epochs)
for METHOD in ae vae crossmodalae contrastiveae; do
    sbatch $DEP -J "emma_clip_match_${METHOD}_64_s0" \
        slurm/train_encfus_80k.sh "$METHOD" 64 0 "$ENC" "$FUS"
    echo "Submitted: emma_clip_match_${METHOD}_64_s0"
done

echo ""
echo "Done — 8 jobs submitted (1 cache builder + 7 compression methods)."
echo "Results will appear in checkpoints/test_results_batched.txt"
echo "Monitor with: squeue -u \$USER"
