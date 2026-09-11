#!/bin/bash
# Re-evaluate every best_server_*.pt checkpoint's TEST-set accuracy (not just
# the val accuracy already stored in each checkpoint). Inference-only, no
# training -- loads GPT-2 once and reuses it across every checkpoint found.
# Overwrites checkpoints/test_results.txt with the complete set, including
# any checkpoints (e.g. LDA seeds 1/2) that were trained but never evaluated.
#
# Usage: sbatch slurm/run_eval_test.sh
#SBATCH --job-name=emma_eval_test
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --error=logs/eval_test_%j.err
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
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
echo "Start:      $(date)"

python3 -u eval_test.py --batch

echo "Done: $(date)"
