#!/bin/bash
# Run once on the HPC login node to install dependencies.
# Usage: bash slurm/setup.sh

set -e

echo "=== EMMA environment setup ==="

# Same modules every other script in this directory loads (train_llava.sh,
# predownload_compression.sh, etc.) -- this file never actually loaded any
# module, just left commented-out generic placeholders, so `pip` wasn't on
# PATH at all until now.
module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1

# Every other script does `source /scratch/user/motahare/emma_env/bin/
# activate` and expects packages to already be there -- this file never
# created or activated that venv at all, it just did `pip install --user`
# straight into the system Python, which the other scripts' activated venv
# would never see. Create it once if missing, then always activate it.
ENV_DIR=/scratch/user/motahare/emma_env
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

pip install \
    transformers datasets numpy

# emma/encoders/mobileclip_encoder.py (--encoder mobileclip everywhere,
# including the compression scripts in this directory) needs open_clip --
# not previously tracked here even though mobileclip runs already
# depended on it being present.
pip install open_clip_torch

# emma/model_compression/__init__.py eagerly imports every submodule,
# including onnx_static.py -- so even train_qat.py / train_pruning_
# recovery.py (which never touch ONNX at all) fail to import the package
# without these. Installed ad-hoc during development, never tracked here.
pip install onnx onnxruntime onnxscript

echo "=== Done. Test with: python3 -c 'import torch; print(torch.cuda.is_available())' ==="
