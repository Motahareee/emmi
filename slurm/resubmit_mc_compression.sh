#!/bin/bash
# Rerun only the 6 MobileCLIP + DistilAE-64 configs that collapsed due to
# L2-normalized embeddings. Fix: normalize=False in mobileclip_encoder.py.
#
# The uncompressed MobileCLIP embed caches are already built and valid
# (the no-compression jobs finished correctly — normalization only affects
# what the AE sees, not the raw CLIP forward pass used for caching).
# BUT: the existing embed caches were built with normalized embeddings,
# so they must be rebuilt. Each cache-builder job does this automatically
# if it doesn't find the cache — so we must delete the old bad caches first.
#
# Run this script from /scratch/user/motahare/emma

cd "$(dirname "$0")/.." 2>/dev/null || true

echo "Removing stale MobileCLIP embed caches (built with normalized embeddings)..."
rm -rf checkpoints/embed_cache_mobileclip_mean_72000_5000_5000
rm -rf checkpoints/embed_cache_mobileclip_concat_72000_5000_5000
rm -rf checkpoints/embed_cache_mobileclip_match_72000_5000_5000
echo "  done."

for FUS in mean concat match; do
    FIRST=$(sbatch --parsable --time=08:00:00 \
        -J "emma_mc_fix_${FUS}_cache" \
        slurm/train_encfus_80k.sh none 0 99 mobileclip "$FUS")
    echo "[mobileclip_$FUS] cache-rebuild job: $FIRST (seed 99, result discarded)"

    DEP="--dependency=afterok:$FIRST"
    for SEED in 0 1 2; do
        sbatch $DEP --time=08:00:00 \
            -J "emma_mc_fix_${FUS}_distilae_64_s${SEED}" \
            slurm/train_encfus_80k.sh distilae 64 "$SEED" mobileclip "$FUS"
    done
done

echo "Submitted 3 cache-rebuild jobs + 9 DistilAE jobs (12 total)."
