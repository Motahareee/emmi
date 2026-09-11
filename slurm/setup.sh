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

echo "=== Done. Test with: python3 -c 'import torch; print(torch.cuda.is_available())' ==="
