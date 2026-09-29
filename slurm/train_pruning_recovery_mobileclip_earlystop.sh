#!/bin/bash
# Same large-data + early-stopping recipe that took CLIP's pruning-
# recovery from 49.65% to 71.80%-82.90% (Findings 26-28), applied to
# MobileCLIP's Conv-channel pruning for the first time -- Findings 20-21
# only ever measured MobileCLIP pruning with NO recovery at all.
#
# Tests two things at once: (1) does recovery even help MobileCLIP's
# Conv-pruning close to the degree it helped CLIP's, and (2) does
# Finding 20's compute-bottleneck argument hold -- i.e. does size keep
# improving with recovery while latency stays flat regardless, because
# the pruned MLP convs were never the compute bottleneck to begin with.
#
# batch-size kept more conservative than CLIP's (64 vs 128) since
# MobileCLIP's Conv activations at 256px may be more memory-hungry per
# image than CLIP's ViT-B/32 sequence at 224px -- adjust up if headroom
# allows.
#
# Run slurm/predownload_compression.sh on the login node FIRST (needs
# the 13500-sample COCO cache).
#
# Usage: sbatch slurm/train_pruning_recovery_mobileclip_earlystop.sh
#   sbatch slurm/train_pruning_recovery_mobileclip_earlystop.sh --ratio 0.1   # other ratios
#SBATCH --job-name=emma_pruning_mc_earlystop
#SBATCH --output=logs/pruning_mc_earlystop_%j.out
#SBATCH --error=logs/pruning_mc_earlystop_%j.err
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

python3 -u train_pruning_recovery_mobileclip.py \
    --ratio 0.3 \
    --n-train 12000 \
    --n-eval 1500 \
    --n-zeroshot 2000 \
    --batch-size 64 \
    --epochs 40 \
    --eval-every 2 \
    --early-stop \
    --lr 1e-4 \
    "$@"

echo "Done: $(date)"
