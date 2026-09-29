#!/bin/bash
# Pre-fetches CLIP, MobileCLIP, and CIFAR-10 on the login node -- compute
# nodes run fully offline (see train_llava.sh's TRANSFORMERS_OFFLINE=1
# etc.), so the GPU jobs below (train_qat_compression.sh,
# train_pruning_recovery.sh) will fail outright if these aren't cached
# first.
#
# Usage: bash slurm/predownload_compression.sh
#
# Not an sbatch script -- run directly on the login node (no GPU needed
# for downloading weights/data).

set -e

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache

cd "$(dirname "$0")/.."

echo "=== Checking open_clip (see setup.sh) ==="
python3 -c "import open_clip" 2>/dev/null || pip install --user open_clip_torch

echo "=== Pre-fetching CLIP + MobileCLIP weights into HF_HOME ==="
python3 -c "
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
print('CLIP cached.')

import open_clip
open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')
print('MobileCLIP cached.')
"

echo "=== Pre-fetching CIFAR-10 (real zero-shot accuracy eval set) ==="
python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='.cifar10_cache', train=False, download=True)
print('CIFAR-10 cached.')
"

echo "=== Pre-fetching COCO fine-tuning images (real image/caption pairs used for distillation) ==="
python3 -c "
from emma.data.coco import _stream_samples
# Matches the scale used in train_qat_compression.sh / train_pruning_recovery.sh --
# see those scripts for why N_TRAIN went from 128 (CPU investigation) to 4000 here.
_stream_samples(5000, offset=0)
print('COCO samples cached.')
"

echo "=== Pre-fetching larger COCO set for the information-theoretic capacity test ==="
python3 -c "
from emma.data.coco import _stream_samples
# train_pruning_recovery_scale.sh -- n_train=12000/n_eval=1500, testing whether
# the ~50-64% pruning-recovery accuracy ceiling is a training-budget artifact
# or intrinsic to the pruned model's remaining capacity.
_stream_samples(13500, offset=0)
print('Large-scale COCO samples cached.')
"

echo "Done: $(date)"
