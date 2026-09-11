#!/bin/bash
# Comprehensive 80K/5K/5K experiment:
#   none x 5 seeds  +  {pca, ae, vae, distilae} x {32, 64, 128} x 5 seeds  =  65 jobs
#
# The first job (none, seed 0) builds the shared CLIP embedding cache;
# all other jobs wait for it via SLURM dependency to avoid 65 jobs racing
# to encode 90K images into the same cache files.
#
# Prerequisite: run precache_coco.py on a login node first (raw image cache).

cd "$(dirname "$0")/.."

# ── Job 1: builds embedding cache, then trains the baseline ──────────────────
FIRST=$(sbatch --parsable -J emma_none_s0 slurm/train_80k.sh none 0 0)
echo "Cache-building job: $FIRST (all others depend on it)"

DEP="--dependency=afterok:$FIRST"

# ── Baseline, remaining seeds ─────────────────────────────────────────────────
for SEED in 1 2 3 4; do
    sbatch $DEP -J "emma_none_s${SEED}" slurm/train_80k.sh none 0 "$SEED"
done

# ── Compression methods x dims x seeds ────────────────────────────────────────
for METHOD in pca ae vae distilae; do
    for DIM in 32 64 128; do
        for SEED in 0 1 2 3 4; do
            sbatch $DEP -J "emma_${METHOD}_${DIM}_s${SEED}" \
                slurm/train_80k.sh "$METHOD" "$DIM" "$SEED"
        done
    done
done

echo "Submitted 1 + 4 + 60 = 65 jobs."
