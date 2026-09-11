#!/bin/bash
#SBATCH --job-name=emma_llava_check
#SBATCH --output=logs/llava_check_%j.out
#SBATCH --error=logs/llava_check_%j.err
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
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

# NOTE: First run needs internet access to download llava-hf/llava-1.5-7b-hf
# (~14GB) into $HF_HOME -- different checkpoint from the one benchmark_llava.py
# uses, so this will download again even if that job already ran.
#
# Requires bitsandbytes for 8-bit loading:
#   pip install --user bitsandbytes
python3 check_llava_capability.py --load-in-8bit

echo "Done: $(date)"
