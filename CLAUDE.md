# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EMMA is a lightweight multimodal AI inference pipeline designed for edge devices. It compresses multimodal inputs on-device and offloads high-level reasoning to a server-side MLLM, optimizing for inference latency on the CMU-MOSEI dataset.

## Environment

The Docker environment (`Dockerfile`) provides:

- **Python 3** with PyTorch, torchvision, torchaudio, Hugging Face `transformers`, and `datasets`
- **Node.js 20** with `@anthropic-ai/claude-code`

```bash
docker build -t emma .
docker run -it --rm -v $(pwd):/workspace emma
```

## Pipeline Architecture

EMMA is a four-stage sequential pipeline. Data flows edge → server:

```
[Raw Input]
    │
    ▼
Stage 1 — Modality Encoders (edge)
    Text   → e.g. lightweight transformer encoder
    Audio  → e.g. spectrogram + CNN or small wav2vec
    Vision → e.g. MobileNet / EfficientNet variant
    │
    ▼
Stage 2 — Cross-Modal Alignment (edge)
    Projects each modality into a shared embedding space.
    Typically a small projection MLP or attention fusion layer.
    │
    ▼
Stage 3 — VAE-Based Compression (edge)
    Encodes the aligned multimodal representation into a
    compact latent vector for efficient transmission.
    The decoder lives server-side.
    │
    ▼
Stage 4 — Server-Side MLLM Reasoning
    Receives the latent, decodes it, and runs a large
    multimodal language model for final predictions.
```

The primary design constraint throughout is **inference latency**: Stages 1–3 must be fast enough to run on edge hardware; Stage 4 may be heavier but network round-trip cost matters.

## Dataset: CMU-MOSEI

CMU-MOSEI provides video, audio, and text for sentiment/emotion analysis. Key details relevant to EMMA:

- Three modalities align at the segment level (word-aligned).
- Standard splits: `train` / `valid` / `test`.
- Labels include sentiment scores and emotion categories.
- Load via the `mmsdk` SDK or Hugging Face `datasets`.

## Primary Metric

**Inference latency** (milliseconds, end-to-end on target edge hardware) is the headline metric. Secondary metrics include accuracy/F1 on CMU-MOSEI tasks and compression ratio (bits transmitted per sample).

## Key Design Decisions

- **VAE bottleneck** (Stage 3) is the main lever for the latency/accuracy trade-off. KL weight (`β`) controls compression tightness.
- **Stage 1 encoders** should be swappable — keep each encoder behind a common interface so they can be replaced with quantized or pruned variants without touching alignment or compression code.
- **Stage 2 alignment** must be trained jointly with Stage 3 compression; the VAE reconstruction target depends on the quality of the shared embedding.
- Stage 4 is server-side and can use a larger model (e.g. a Hugging Face MLLM); keep it decoupled from the edge stages via a clean latent-vector interface.
