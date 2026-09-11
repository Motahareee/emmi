#!/bin/bash
# Complete EMMA latency benchmark — single CPU thread, edge simulation.
# Covers: encoders, fusion, compression, server GPT-2, transmission.
# Run with:   sbatch slurm/run_latency.sh
#
#SBATCH --job-name=emma_latency
#SBATCH --output=logs/latency_%j.out
#SBATCH --error=logs/latency_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
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

echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "CPUs:    $SLURM_CPUS_PER_TASK  (single-thread edge simulation)"
echo ""

python3 benchmark_latency.py --n-runs 200 --warmup 30 --threads 1

echo ""
echo "Done: $(date)"
