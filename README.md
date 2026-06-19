# EMMA — Edge Multimodal Model Architecture

EMMA is a lightweight multimodal AI inference pipeline designed for edge devices. It encodes multimodal inputs on-device and offloads high-level reasoning to a server-side LLM, optimizing for inference latency.

## Overview

```
[Raw Input]
     │
     ▼
Stage 1 — Modality Encoders (edge)
     Text   → DistilBERT-base (frozen) + linear projection → 256-dim
     Audio  → Whisper-small encoder (frozen) + linear projection → 256-dim
     │
     ▼
Stage 2 — Cross-Modal Alignment (edge)
     Fuses text + audio into a shared 256-dim embedding
     │
     ▼
Stage 3 — VAE Compression (edge, optional)
     Compresses the shared embedding for efficient transmission
     │
     ▼
Stage 4 — Server-Side LLM Reasoning
     Soft token injection → frozen LLM → task heads
     Sentiment regression (MSE) + Emotion classification (BCE)
```

## Dataset

**CMU-MOSEI** via `cairocode/cmu_mosei_wav` (HuggingFace Hub)

| Split | Samples |
|-------|---------|
| Train | 3,597   |
| Valid | 742     |
| Test  | 906     |

Labels: continuous sentiment score + 6 emotion scores (happy, sad, anger, surprise, disgust, fear)

## Results (no compression, CPU, GPT-2 server)

| Metric | Value |
|--------|-------|
| Sentiment MAE | 0.668 |
| Sentiment Accuracy | 55.5% |
| Emotion Accuracy | 83.1% |
| Edge latency (median) | 921ms |
| Server latency (median) | 57ms |
| Total latency (median) | 986ms |

## Project Structure

```
/workspace/
│
├── train.py              # Main training script (all stages)
├── benchmark.py          # Latency + task metrics evaluation
├── Dockerfile            # Container definition
│
├── emma/                 # Core library
│   ├── config.py         # All hyperparameters as dataclasses
│   │
│   ├── encoders/         # Stage 1 — per-modality encoders
│   │   ├── text_encoder.py       DistilBERT + projection head
│   │   ├── audio_encoder.py      Whisper-small + projection head
│   │   ├── vision_encoder.py     MobileNetV3 (reserved for future vision)
│   │   └── feature_projector.py  Linear projector for pre-extracted features
│   │
│   ├── alignment/        # Stage 2 — cross-modal fusion
│   │   └── cross_modal.py        Projects modalities to shared 256-dim space
│   │
│   ├── compression/      # Stage 3 — VAE (architecture ready, training TBD)
│   │   └── vae.py                Beta-VAE encoder/decoder
│   │
│   ├── data/             # Dataset
│   │   └── mosei.py              CMU-MOSEI loader (HuggingFace + mmsdk)
│   │
│   ├── server/           # Stage 4 — server-side LLM
│   │   └── pipeline.py           Soft token injection + multi-task heads
│   │
│   └── pipeline.py       # EdgePipeline — wires all edge stages together
│
├── checkpoints/          # Saved model weights (not in git)
│   ├── best_edge.pt              Best edge stage checkpoint
│   ├── best_server.pt            Best server stage checkpoint
│   └── embed_cache/              Cached fused embeddings
│
└── mmsdk_cache/          # Visual feature cache (CMU-MOSEI Facet 4.2)
```

## Training

```bash
python3 train.py
```

Two stages:
1. **Edge** — trains projection heads + alignment jointly (10 epochs)
2. **Server** — trains server projection + task heads on cached embeddings (10 epochs)

Resumes automatically from checkpoint if `checkpoints/best_edge.pt` exists.

## Benchmark

```bash
python3 benchmark.py
```

Reports task metrics (MAE, accuracy) and latency (median, p95) broken down by edge vs server.

## Setup

```bash
docker build -t emma .
docker run -it --rm -v $(pwd):/workspace emma
```

## Server LLM Scenarios

Three ablation scenarios configured via `ServerConfig.scenario`:

| Scenario | Model | Description |
|----------|-------|-------------|
| `plain_llm` | Mistral-7B | Text-only baseline |
| `llava` | LLaVA-1.5-7B | LLM pre-trained on visual soft tokens |
| `qwen_audio` | Qwen2-Audio-7B | LLM pre-trained on audio soft tokens |

Set `DEBUG=True` in `train.py` to use GPT-2 instead (no GPU needed).
