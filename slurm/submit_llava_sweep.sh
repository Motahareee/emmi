#!/bin/bash
# Full LLaVA-server sweep: 8 compressors x 2 encoders, match fusion, d=64 --
# the LLaVA counterpart to the paper's full GPT-2 headline table.
# Skips cells already completed. Safe to re-run: sbatch just queues a new
# job for anything not explicitly skipped below.
#
# Usage: bash slurm/submit_llava_sweep.sh
#
# --encoder overrides train_llava.sh's own hardcoded "--encoder mobileclip"
# via argparse's normal last-value-wins behavior -- no need to touch
# train_llava.sh itself.

cd "$(dirname "$0")/.."

# (encoder, compression) pairs already completed -- see logs/llava_train_*
# for the underlying results (mobileclip+contrastiveae: 98.19% test acc,
# mobileclip+crossmodalae: 90.64% test acc).
#
# PCA (both encoders) is skipped for now: mobileclip+pca produces a NaN
# gradient sometime after batch 3 of epoch 1 (bit-identical, healthy first 3
# batches every time -- consistent with CUDA backward-pass non-determinism
# tipping a numerically marginal setup into fp16 overflow), and it failed
# the same way on 3 separate attempts including with the NaN-guard active.
# clip+pca hasn't been tested but shares the same fragile compressor, so
# it's held back too pending a real fix (tighter grad clip / lower LR /
# deterministic algorithms) rather than spending more cluster time guessing.
SKIP=(
    "mobileclip contrastiveae"
    "mobileclip crossmodalae"
    "mobileclip pca"
    "clip pca"
)

should_skip() {
    local enc="$1" comp="$2"
    for pair in "${SKIP[@]}"; do
        if [ "$pair" == "$enc $comp" ]; then
            return 0
        fi
    done
    return 1
}

for ENCODER in mobileclip clip; do
    for COMPRESSION in none pca ae vae lda blockpca contrastiveae crossmodalae; do
        if should_skip "$ENCODER" "$COMPRESSION"; then
            echo "Skipping ${ENCODER}+${COMPRESSION} -- already completed"
            continue
        fi
        TAG="emma_llava_${ENCODER}_${COMPRESSION}"
        sbatch -J "$TAG" --time=16:00:00 --partition=gpu --gres=gpu:rtx:1 \
            slurm/train_llava.sh --load-in-8bit --patience 5 \
            --encoder "$ENCODER" --compression "$COMPRESSION"
        echo "Submitted: $TAG"
    done
done

echo "Done submitting sweep."
