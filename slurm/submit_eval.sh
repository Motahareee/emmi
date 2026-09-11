#!/bin/bash
# Submit all evaluation batches as independent SLURM jobs.
# Each job evaluates 3 checkpoints, needs ~32G RAM, ~20min.
# Usage: bash slurm/submit_eval.sh
#        bash slurm/submit_eval.sh 0 3    # only submit batches 0,1,2,3

cd "$(dirname "$0")/.."

START=${1:-0}
END=${2:-15}

echo "Submitting eval batches $START to $END ..."
echo ""

for i in $(seq "$START" "$END"); do
    JOB_ID=$(sbatch --parsable -J "eval_b${i}" slurm/eval_batch.sh "$i")
    echo "  Batch $i → job $JOB_ID"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
echo "Results will be in: checkpoints/test_results_batched.txt"
