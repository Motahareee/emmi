#!/bin/bash
# Run once on the HPC login node to install dependencies.
# Usage: bash slurm/setup.sh

set -e

echo "=== EMMA environment setup ==="

# Load common HPC modules — adjust names for your cluster
# module load python/3.11
# module load cuda/11.8

# Install into user space (no sudo needed)
pip install --user \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

pip install --user \
    transformers datasets numpy

# emma/data/avqa.py (audio-visual QA extension) needs these -- soundfile
# reads the ffmpeg-extracted wav clips, yt-dlp downloads them in the first
# place. Both are imported unconditionally (even on a cache hit), so they
# have to be present in this env before any AVQA script runs, not just on
# the login node -- see slurm/predownload_avqa.sh.
pip install --user \
    soundfile yt-dlp

echo "=== Done. Test with: python3 -c 'import torch; print(torch.cuda.is_available())' ==="
