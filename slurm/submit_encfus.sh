#!/bin/bash
# Encoder/fusion screening experiment — 72K/5K/5K splits, 3 seeds.
#
#   (encoder, fusion) combos:  clip+concat, clip+match,
#                              mobileclip+mean, mobileclip+concat, mobileclip+match
#   (clip+mean is the finished comprehensive experiment — not repeated)
#
#   per combo: {none, distilae 64} x 3 seeds = 6 jobs  →  5 x 6 = 30 jobs
#
# Each combo's first job (none, seed 0) builds that combo's embed cache;
# the other 5 jobs of the combo depend on it. Different combos run freely
# in parallel (separate cache dirs), but their 5 cache-building jobs all
# read the raw pkls — acceptable contention (5 jobs, not 22).
#
# PREREQUISITE (login node, once):
#   source /scratch/user/motahare/emma_env/bin/activate
#   pip install open_clip_torch
#   export HF_HOME=/scratch/user/motahare/hf_cache
#   python3 -c "import open_clip; open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')"

cd "$(dirname "$0")/.."

for COMBO in "clip concat" "clip match" "mobileclip mean" "mobileclip concat" "mobileclip match"; do
    set -- $COMBO
    ENC=$1; FUS=$2
    TAG="${ENC}_${FUS}"

    FIRST=$(sbatch --parsable -J "emma_${TAG}_none_s0" \
        slurm/train_encfus_80k.sh none 0 0 "$ENC" "$FUS")
    echo "[$TAG] cache-building job: $FIRST"
    DEP="--dependency=afterok:$FIRST"

    for SEED in 1 2; do
        sbatch $DEP -J "emma_${TAG}_none_s${SEED}" \
            slurm/train_encfus_80k.sh none 0 "$SEED" "$ENC" "$FUS"
    done
    for SEED in 0 1 2; do
        sbatch $DEP -J "emma_${TAG}_distilae_64_s${SEED}" \
            slurm/train_encfus_80k.sh distilae 64 "$SEED" "$ENC" "$FUS"
    done
done

echo "Submitted 30 encoder/fusion jobs (5 combos x 6)."
