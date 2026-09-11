#!/bin/bash
# Submit all remaining 80K experiments.
# PCA 64 and AE 64 already done — commented out.
cd "$(dirname "$0")/.."

# Baseline
sbatch -J emma_none_80k        slurm/train_80k.sh none      0

# 64-dim (main table)
# sbatch -J emma_pca_64_80k    slurm/train_80k.sh pca       64   # done: 94.7%
# sbatch -J emma_ae_64_80k     slurm/train_80k.sh ae        64   # done: 95.1%
sbatch -J emma_vae_64_80k      slurm/train_80k.sh vae       64
sbatch -J emma_distilae_64_80k slurm/train_80k.sh distilae  64
sbatch -J emma_distvarae_64_80k slurm/train_80k.sh distvarae 64

# 128-dim
sbatch -J emma_pca_128_80k     slurm/train_80k.sh pca       128
sbatch -J emma_ae_128_80k      slurm/train_80k.sh ae        128
sbatch -J emma_vae_128_80k     slurm/train_80k.sh vae       128
sbatch -J emma_distilae_128_80k slurm/train_80k.sh distilae 128

# 32-dim
sbatch -J emma_pca_32_80k      slurm/train_80k.sh pca       32
sbatch -J emma_ae_32_80k       slurm/train_80k.sh ae        32
sbatch -J emma_vae_32_80k      slurm/train_80k.sh vae       32
sbatch -J emma_distilae_32_80k slurm/train_80k.sh distilae  32

echo "All 80K jobs submitted."
