# EMMI

**Paper:** [EMMI: Edge Multi-Modal Intelligence for Communication-Efficient MLLM Inference via Fused Representation Compression](https://arxiv.org/abs/2609.11058) — Motahare Mounesan, Irfan Khan

EMMI is a lightweight multimodal AI inference pipeline for edge-server split deployment. It encodes and compresses multimodal inputs on-device, transmits a compact latent representation, and offloads reasoning to a server-side language model — optimizing for inference latency and communication payload while preserving task accuracy.

The task is **binary image-text matching** on MS-COCO: given an image and a caption, does the caption describe the image?

> **Note on naming:** the Python package and all imports (`emma/`, `from emma.config import ...`, etc.) still use the original codename `emma` — only the project's public name has moved to EMMI.

## Overview

```
[Raw Input: image + caption]
     │
     ▼
Stage 1 — Modality Encoders (edge, frozen)
     Text + Image  → CLIP (ViT-B/32) or MobileCLIP2-S0  → 512-dim each
     │
     ▼
Stage 2 — Cross-Modal Alignment (edge)
     Fuses text + image embeddings:
       mean   → 512-dim
       concat → 1024-dim
       match  → 2048-dim  [t ; v ; |t-v| ; t⊙v]
     │
     ▼
Stage 3 — Compression (edge, optional)
     Compresses the fused embedding (typically to 64-dim) for transmission.
     Seven methods, spanning closed-form and learned, label-free and
     label-supervised:
       PCA, LDA          — closed-form, no gradient training
       BlockPCA          — closed-form, per-structural-block PCA (match fusion only)
       AE, VAE           — learned, unsupervised (MSE / β-VAE reconstruction)
       CrossModalAE      — learned, task-agnostic (InfoNCE on natural image-text pairing, no labels)
       ContrastiveAE     — learned, task-aware (MSE + supervised contrastive loss on match labels)
     │
     ▼
Stage 4 — Server-Side LLM Reasoning
     Soft-token injection → frozen LLM backbone → binary match head (BCE)
     Backbone is swappable: GPT-2 (fast iteration) or a full MLLM
     (LLaVA-1.5-7B, Qwen2-Audio-7B, Mistral-7B) — see "Server LLM Scenarios" below.
```

Only the compression stage (when learned) and the server-side projection + match head are trained; the edge encoders and the server LLM backbone are frozen throughout.

## Dataset

**MS-COCO** image-text matching, loaded via `emma/data/coco.py`. Streamed from Hugging Face `datasets`, split into non-overlapping train/valid/test windows.

Default splits: 5,000 / 500 / 500 (`--n-train --n-valid --n-test`). Headline paper results use the larger "72k" split: 72,000 / 5,000 / 5,000.

## Results (server: GPT-2, 64-dim compression, 256B payload = 32× reduction)

| Method | Supervision | CLIP + match fusion | MobileCLIP + match fusion |
|---|---|---|---|
| None (uncompressed, 2048-dim) | — | 97.92% | 98.41% |
| PCA-64 | none | 96.88% | 57.35% |
| AE-64 | none | 93.40% | 52.01% |
| VAE-64 | none | 96.75% | 69.87% |
| BlockPCA-64 | none | 97.12% | 97.85% |
| LDA-64 | labels | 97.78% | 98.33% |
| CrossModalAE-64 (task-agnostic) | pairs only | 94.09% | 90.30% |
| ContrastiveAE-64 (task-aware) | labels | 98.08% | 98.32% |

MobileCLIP embeddings have higher intrinsic dimensionality than CLIP's, which is why generic compressors (PCA/AE/VAE) collapse on MobileCLIP but not CLIP — structure-aware methods (LDA, BlockPCA, ContrastiveAE) are robust to this because they exploit task- or block-level structure rather than raw variance alone.

## Extension: full-MLLM server backbone

`train_llava.py` swaps GPT-2 for LLaVA-1.5-7B's `language_model` backbone (vision tower discarded, EMMI's own soft tokens injected instead), reusing an already-trained compressor checkpoint rather than retraining it. Most (compressor, encoder) combinations match their GPT-2 accuracy within a few points; a subset (some AE/LDA/VAE/PCA cells on MobileCLIP+match) hit a numerical instability during LLaVA training that's under active investigation — see `slurm/test_llava_*.sh` for the diagnostic scripts.

## Project Structure

```
/workspace/
│
├── train.py                  # Main training script: edge encode → compression → server (GPT-2 backbone)
├── train_llava.py            # Server training against a full MLLM backbone (loads a pretrained compressor)
├── benchmark.py               # Task metrics evaluation
├── benchmark_latency.py       # Per-stage latency benchmark (encoders, fusion, compression, server GPT-2/LLaVA)
├── benchmark_llava.py         # End-to-end LLaVA-NeXT latency benchmark
├── check_llava_capability.py  # Smoke test: does the LLaVA backbone load and run a forward pass
├── check_ckpts.py             # List validation accuracy across saved checkpoints
├── eval_batch.py / eval_paper.py / eval_quantization.py / eval_test.py   # Test-set evaluation scripts
├── precache_coco.py           # Pre-build the COCO embedding cache
├── Dockerfile
│
├── emma/                      # Core library
│   ├── config.py                     All hyperparameters as dataclasses
│   ├── encoders/                     Stage 1 — per-modality encoders
│   │   ├── text_encoder.py             CLIP text encoder
│   │   ├── image_encoder.py            CLIP image encoder
│   │   └── mobileclip_encoder.py       MobileCLIP2-S0 text + image encoders (open_clip)
│   ├── alignment/
│   │   └── cross_modal.py            Stage 2 — mean / concat / match fusion
│   ├── compression/                  Stage 3 — compression methods
│   │   ├── vae.py                      β-VAE
│   │   ├── autoencoder.py              AE, ContrastiveAE (SupCon), CrossModalAE (InfoNCE)
│   │   ├── pca.py                      Closed-form PCA
│   │   ├── lda.py                      LDA-PCA hybrid (task-aware, closed-form)
│   │   └── block_pca.py                Per-structural-block PCA (match fusion only)
│   ├── data/
│   │   └── coco.py                   MS-COCO image-text matching loader
│   ├── server/
│   │   └── pipeline.py               Stage 4 — soft-token injection + binary match head;
│   │                                   GPT-2 / LLaVA / Qwen2-Audio / Mistral backbones
│   └── pipeline.py                   EdgePipeline — wires the frozen edge stages together
│
├── reports/                    # Paper source: submission.tex, related_work.tex, references.bib,
│                                 proposed_solution.tex, results_draft.tex, figures, analysis scripts
├── slurm/                     # Cluster job scripts: training sweeps, latency benchmarks, evaluation, diagnostics
└── checkpoints/                # Saved model weights + embedding caches (not in git)
```

## Training

```bash
python3 train.py \
    --compression contrastiveae \
    --encoder mobileclip \
    --fusion match \
    --n-train 72000 --n-valid 5000 --n-test 5000
```

Two stages:
1. **Compression** — skipped for `--compression none`; closed-form fit for `pca`/`lda`/`blockpca`; up to 50 epochs of gradient training for `ae`/`vae`/`contrastiveae`/`crossmodalae`.
2. **Server** — 20 epochs by default (`--epochs`), early stopping (patience 10 by default, `--patience`), training only the projection MLP + match head against a frozen LLM.

Use `--no-debug` to use a full LLM instead of GPT-2 — note that `train.py` hardcodes this to the `plain_llm` scenario (Mistral-7B) with no CLI flag to pick a different one; use `train_llava.py` for the `llava` scenario specifically (see below).

For a full MLLM backbone specifically, use `train_llava.py` instead — it reuses an already-trained compressor checkpoint rather than retraining it:

```bash
python3 train_llava.py --compression contrastiveae --encoder mobileclip --load-in-8bit
```

## Benchmark

```bash
python3 benchmark_latency.py
```

Reports per-stage latency: encoder forward pass, fusion, compression encode, server inference (GPT-2 and LLaVA, GPU), and transmission latency across several bandwidth profiles.

## Setup

```bash
docker build -t emma .
docker run -it --rm -v $(pwd):/workspace emma
```

## Server LLM Scenarios

| Backbone | Model | Notes |
|----------|-------|-------|
| GPT-2 | `gpt2` | Produces every result in the paper's headline table (`DEBUG=True` in `train.py`, the default) |
| `llava` | LLaVA-1.5-7B | See "Extension: full-MLLM server backbone" above; run via `train_llava.py` |

`train.py --no-debug` does not support selecting `llava` via CLI — use `train_llava.py` instead, which also supports loading the backbone in 8-bit (`--load-in-8bit`) or fp32 (`--llm-dtype float32`) for memory-constrained GPUs.
