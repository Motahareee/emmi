#!/bin/bash
# Submit all EMMA experiments to SLURM.
# Usage: bash slurm/submit_all.sh
#
# Dependency chain:
#   encode → [none, ae, vae] run in parallel

set -e
mkdir -p logs

# Step 1 — encode CLIP embeddings (runs once, all others depend on it)
# If embed_cache already exists on the HPC, skip this and remove --dependency below.
ENC_JID=$(sbatch --parsable slurm/train_none.sh)
echo "Submitted encode+none job: $ENC_JID"

# Steps 2a/2b — AE and VAE run after encoding is done
AE_JID=$(sbatch --parsable --dependency=afterok:$ENC_JID slurm/train_ae.sh)
echo "Submitted AE job:          $AE_JID (after $ENC_JID)"

VAE_JID=$(sbatch --parsable --dependency=afterok:$ENC_JID slurm/train_vae.sh)
echo "Submitted VAE job:         $VAE_JID (after $ENC_JID)"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      logs/"
