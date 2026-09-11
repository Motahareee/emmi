#!/bin/bash
# Genuine full-fp32 test for mobileclip+ae -- the real precision-isolation
# test the earlier "fp16" run should have been but wasn't: that run still
# force-cast the backbone to float16 (a fix for an unrelated dtype-mismatch
# crash), so it never actually removed fp16's ~65504 dynamic-range ceiling.
# This run does, via --llm-dtype float32.
#
# Needs A40 (48GB) -- fp32 weights alone are ~26GB, won't fit on RTX (24GB).
#
# Usage: sbatch slurm/test_llava_ae_fp32.sh
#SBATCH --job-name=emma_llava_mobileclip_ae_fp32
#SBATCH --output=logs/llava_train_%j.out
#SBATCH --error=logs/llava_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --partition=gpu-a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"

python3 -u train_llava.py \
    --encoder mobileclip \
    --fusion match \
    --compression ae \
    --n-train 72000 \
    --n-valid 5000 \
    --n-test 5000 \
    --batch-size 64 \
    --patience 5 \
    --llm-dtype float32 \
    --debug-forward \
    --checkpoint-dir checkpoints

echo "Done: $(date)"
