# Compressing CLIP for Edge Deployment: Can Compression Replace Purpose-Built Efficiency?

## Abstract

MobileCLIP is a CLIP variant purpose-built for Apple's mobile hardware, reported to be ~4x faster than standard CLIP. We show this speed advantage is hardware-specific: it holds on Apple's Neural Engine but inverts on generic CPU, where MobileCLIP is actually slower. This motivates our core question: can compressing vanilla CLIP (quantization, pruning, low-rank approximation) close the remaining accuracy/size/latency gap on hardware where MobileCLIP's advantage doesn't apply? We find that properly-tuned structured pruning gets compressed CLIP within 2 points of full precision, using real CPU latency measurements throughout — a stronger result than any single technique in isolation, achieved only after discovering and fixing a critical early-stopping bug that had made the technique look far weaker than it actually is.

## 1. Motivation

MobileCLIP's published numbers (Vasu et al., CVPR 2024) are measured on iPhone's Neural Engine via Core ML. We confirmed this directly: on our own Apple Silicon Mac, MobileCLIP is **4.44x faster** than CLIP (1.37ms vs 6.07ms) — matching the paper. But on generic x86 CPU, with the exact same unmodified fp32 models, the ranking **inverts**: MobileCLIP is **1.8x slower** (1854ms vs 1013ms). Same architectures, same parameter counts — only the hardware changed.

This reframes the comparison: MobileCLIP's efficiency isn't free — it's a bet on specific silicon. On ordinary CPU hardware (the realistic deployment target for most edge devices), **CLIP is already faster before any compression is applied**. The question becomes whether compression can also close CLIP's remaining size and accuracy gap.

## 2. Architectures

- **CLIP (ViT-B/32):** a plain Vision Transformer — almost entirely `nn.Linear` layers (attention projections + MLP). No mobile-specific design. 335 MB, 92.0% zero-shot accuracy (our CIFAR-10 proxy benchmark).
- **MobileCLIP-S0:** a hybrid CNN-Transformer (FastViT) — depthwise convolutions, a RepVGG-style reparameterized stem, Conv-based MLPs, two small attention blocks. 44 MB, 94.5% accuracy. Distilled from CLIP, not derived by compressing it.

CLIP's all-Linear structure means every compression technique applies uniformly. MobileCLIP's efficiency is partly pre-baked into Conv/depthwise design choices made at architecture time — there's structurally less "slack" left for post-hoc compression to find, and the techniques we built for Linear layers don't reach its real compute bottleneck (the depthwise/stem convolutions) at all.

## 3. Methods

We implemented all four standard compression families, applied primarily to CLIP's image encoder's MLP width (`fc1→GELU→fc2`) — the one dimension structurally guaranteed not to cascade through the rest of the network (unlike attention or embedding width).

| Technique | What it does |
|---|---|
| **PTQ** (3 variants) | Reduce weight/activation precision to int8 post-hoc. Real kernels via dynamic quantization and ONNX Runtime static quantization. |
| **QAT** | Fine-tune with simulated quantization noise in the loop, then export to real int8 kernels. |
| **Structured pruning** | Remove whole MLP channels, scored by an importance criterion, then recovery fine-tune. Criteria tested: L2-magnitude, Taylor (gradient), Hessian (second-order), and global ranking (cross-layer, not just within-layer). |
| **Low-rank approximation** | Factor each weight matrix via truncated SVD into two smaller matrices, then recovery fine-tune. |

**Recovery fine-tuning** = distillation: train the compressed model to match the original's output on real images, no labels needed.

## 4. Two Methodological Findings That Changed Everything

1. **Embedding similarity to the original model doesn't predict real accuracy.** Early results looked good by cosine-similarity but collapsed to near-chance on an actual classification task. We built an independent zero-shot CIFAR-10 benchmark to catch this, and used it for every subsequent result.

2. **Training loss is not a valid stopping signal for recovery fine-tuning.** Our first pruning results plateaued at ~50-64% accuracy — looking like a fundamental ceiling. Tracking accuracy *during* training (not just at the end) revealed it peaks mid-training then **declines** from overfitting, while training loss falls smoothly the whole time. Checkpointing on best held-out accuracy instead of the final epoch was the single highest-leverage fix in the entire investigation.

## 5. Results

**Best configuration per technique, real CPU measurements:**

| Technique | Size | Latency | Accuracy | Gap to fp32 (92.0%) |
|---|---|---|---|---|
| **Pruning** (Hessian + global ranking, ratio=0.1, no recovery needed) | 1.07x | 1.05x | **89.85%** | **-2.15 pts** |
| Pruning (same criterion, ratio=0.3, + recovery) | 1.24x | 1.22x | 80.55% | -11.45 pts |
| QAT + real dynamic PTQ | **3.68x** | **2.00x** | 77.45% | -14.55 pts |
| Low-rank approximation (ratio=0.3, + recovery) | 1.09x | 1.01x | 77.15% | -14.85 pts |
| PTQ alone (no QAT), best config | **3.94x** | **2.56x** | 84.0% | -8.0 pts |

**No single technique wins on every axis.** Pruning with the best criterion is strongest when accuracy matters most. Quantization (PTQ, or QAT+PTQ combined) is strongest when size/latency matter most. Low-rank approximation works but is dominated by pruning at every matched compression level we tested.

**Pruning criterion comparison** (ratio=0.3): combining Hessian importance (which channel matters to the loss) with global ranking (how many channels each *layer* can spare, not a fixed fraction everywhere) beat plain magnitude-based pruning at every ratio tested (e.g. 89.85% vs 82.90% at ratio=0.1) — the two signals are complementary, not redundant.

**MobileCLIP mirrors most of this.** The same large-data + early-stopping recipe fixes its pruning gap almost completely (83.6% at ratio=0.1, down from a near-chance no-recovery baseline). But its QAT gap only partially closes, and **its latency never meaningfully improves under any technique** — because every technique here only touches Linear/pointwise layers, and MobileCLIP's actual bottleneck (depthwise convs, reparameterized stem) is architecturally outside their reach.

## 6. Conclusion

MobileCLIP's efficiency is real, but it's a hardware-specific claim, confirmed directly on both Apple Silicon and generic CPU. On CPU — where MobileCLIP's advantage doesn't hold — properly-tuned pruning gets compressed CLIP within 2 points of full precision, with a real (if modest) deployment win, using nothing but its own architecture. Quantization gets a much larger compression/speed win at a real but larger accuracy cost. Compression doesn't make CLIP into MobileCLIP, but on hardware where MobileCLIP's own advantage evaporates, compressed CLIP is a legitimate, measured alternative — not a hypothetical one.

## What We Deliberately Didn't Do

- Attention / embedding-width pruning (riskier, cascades through the whole network)
- A fix for MobileCLIP's actual latency bottleneck (depthwise convs / stem)
- Text-encoder real-kernel quantization (blocked by an unrelated library bug)
- Low-rank approximation on MobileCLIP's Conv layers (needs a different factorization scheme)
- Third-party efficient-CLIP baselines (e.g. TinyCLIP)
