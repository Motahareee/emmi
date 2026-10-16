"""
Diagnoses why MobileCLIP's raw image-embedding block ("v") has ~80x less
variance than its text block ("t") in the match-fused cache, while for CLIP
it's the opposite (v is the highest-variance block). Both encoders are
called with normalize=False (see emma/encoders/mobileclip_encoder.py /
emma/encoders/clip_encoder.py), so this isn't L2-normalization flattening
variance -- something about the raw unnormalized outputs themselves differs.

Checks, per encoder, on the cached train_fused.pt image block:
  - per-sample L2 norm distribution (mean/std/min/max)
  - mean pairwise cosine similarity across a random sample of vectors
    (near 1.0 would mean the model is mapping different images to nearly
    the same direction -- a real representation-collapse signal, not just
    a scale artifact)
  - per-dimension variance spread (how concentrated vs. spread the low
    total variance is across the 512 dims)
"""

import torch
import numpy as np


def inspect(cache_dir, name, d_block=512, n_pairs=2000, seed=0):
    X = torch.load(f"{cache_dir}/train_fused.pt", weights_only=True).numpy().astype(np.float32)
    t = X[:, :d_block]
    v = X[:, d_block:2 * d_block]

    print(f"\n=== {name} ===")
    for label, block in (("t", t), ("v", v)):
        norms = np.linalg.norm(block, axis=1)
        print(f"  {label} norms: mean={norms.mean():.4f} std={norms.std():.4f} "
              f"min={norms.min():.4f} max={norms.max():.4f}")

    rng = np.random.RandomState(seed)
    idx_a = rng.randint(0, v.shape[0], n_pairs)
    idx_b = rng.randint(0, v.shape[0], n_pairs)
    a, b = v[idx_a], v[idx_b]
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    print(f"  v pairwise cosine similarity (random pairs): "
          f"mean={cos.mean():.4f} std={cos.std():.4f} min={cos.min():.4f} max={cos.max():.4f}")

    per_dim_var = v.var(axis=0)
    print(f"  v per-dim variance: mean={per_dim_var.mean():.6f} "
          f"max={per_dim_var.max():.6f} (dim {per_dim_var.argmax()}) "
          f"top5_share={np.sort(per_dim_var)[-5:].sum() / per_dim_var.sum() * 100:.1f}%")


if __name__ == "__main__":
    inspect("checkpoints/embed_cache_mobileclip_match_72000_5000_5000", "MobileCLIP + match")
    inspect("checkpoints/embed_cache_match_72000_5000_5000", "CLIP + match")
