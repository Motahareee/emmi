#!/bin/bash
# Low-rank approximation + recovery, the same large-data + early-stopping
# recipe that took pruning from 49.65% to 71.80%-82.90% (Findings 26-28).
# Direct comparison point against pruning's ratio=0.3 result (71.80%
# accuracy, 270.3MB, 1.22x latency speedup) -- the fourth and final
# original compression technique, now tested at comparable scale.
#
# ratio=0.3 gives rank=538 (full_rank=768 for ViT-B/32's 768<->3072 MLP),
# below the 614 breakeven threshold, so this DOES compress (unlike
# smaller ratios -- see lowrank.py's docstring).
#
# Run slurm/predownload_compression.sh on the login node FIRST if you
# haven't already (needs the 13500-sample COCO cache).
#
# Usage: sbatch slurm/train_lowrank_recovery_earlystop.sh
#   sbatch slurm/train_lowrank_recovery_earlystop.sh --ratio 0.5   # other ratios
#SBATCH --job-name=emma_lowrank_earlystop
#SBATCH --output=logs/lowrank_earlystop_%j.out
#SBATCH --error=logs/lowrank_earlystop_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
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

python3 -u train_lowrank_recovery.py \
    --ratio 0.3 \
    --n-train 12000 \
    --n-eval 1500 \
    --n-zeroshot 2000 \
    --batch-size 128 \
    --epochs 40 \
    --eval-every 2 \
    --early-stop \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
