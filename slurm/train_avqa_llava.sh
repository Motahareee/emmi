#!/bin/bash
# Train the AVQA extension with the real "llava" server scenario (GPU),
# using close to the full MUSIC-AVQA video pool (~1,035 videos with real
# YouTube ids/timestamps -- see emma/data/avqa.py's module docstring for
# why it's capped there, not the full academic benchmark).
#
# Run slurm/predownload_avqa.sh on the login node FIRST -- this job runs
# fully offline (TRANSFORMERS_OFFLINE etc. below) and will fail outright
# if the llava weights and video clips aren't already cached.
#
# Usage:
#   sbatch slurm/train_avqa_llava.sh                 # fp16
#   sbatch slurm/train_avqa_llava.sh --load-in-8bit   # 8-bit via bitsandbytes
#SBATCH --job-name=emma_avqa_llava_train
#SBATCH --output=logs/avqa_llava_train_%j.out
#SBATCH --error=logs/avqa_llava_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
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

# 830/100/100 videos is close to the full ~1,035-video pool (some margin
# for the fraction that were unavailable at predownload time -- see
# predownload_avqa.sh's ok/failed counts). Lower these if that run cached
# fewer than ~1,030 clips total.
python3 -u train_avqa.py \
    --encoder clip \
    --scenario llava \
    --n-train-videos 830 \
    --n-valid-videos 100 \
    --n-test-videos 100 \
    --batch-size 8 \
    --epochs 15 \
    --checkpoint-dir checkpoints \
    "$@"

echo "Done: $(date)"
