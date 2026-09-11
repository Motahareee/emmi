#!/bin/bash
#SBATCH --job-name=emma_llava_bench
#SBATCH --output=logs/llava_bench_%j.out
#SBATCH --error=logs/llava_bench_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a40
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cststm@gmail.com

set -e
mkdir -p logs

module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.1.1
source /scratch/user/motahare/emma_env/bin/activate

export HF_HOME=/scratch/user/motahare/hf_cache

cd "$SLURM_SUBMIT_DIR"

echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start:      $(date)"

# NOTE: First run requires internet access to download LLaVA-1.5-7B (~14GB).
# If the model is already cached at $HF_HOME, set TRANSFORMERS_OFFLINE=1.
# To download: run this script WITHOUT the OFFLINE flag first, then re-run with it.

python3 benchmark_llava.py

echo "Done: $(date)"
