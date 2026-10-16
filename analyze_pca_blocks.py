"""
Checks whether global PCA-64 on match-fused embeddings systematically
under-represents the |t-v| and t*v blocks relative to their own variance,
compared to the raw t/v blocks -- the hypothesized reason BlockPCA recovers
accuracy that global PCA loses on MobileCLIP but not on CLIP.

Reads already-cached train_fused.pt embeddings (no retraining); refits a
fresh global PCA-64 directly (cheap -- just one SVD) rather than trusting
checkpoints/pca_64_s0.npz on disk, since that file's name doesn't encode
encoder/fusion and gets overwritten across sweep runs that share the same
--checkpoint-dir.
"""

import torch
import numpy as np


def analyze(cache_dir, name, k=64, d_block=512):
    X = torch.load(f"{cache_dir}/train_fused.pt", weights_only=True).numpy().astype(np.float32)
    blocks = {
        "t":     (0, d_block),
        "v":     (d_block, 2 * d_block),
        "|t-v|": (2 * d_block, 3 * d_block),
        "t*v":   (3 * d_block, 4 * d_block),
    }

    total_var = {name_: X[:, s:e].var(axis=0).sum() for name_, (s, e) in blocks.items()}
    grand_total = sum(total_var.values())

    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    eigvals = (S[:k] ** 2) / (X.shape[0] - 1)

    print(f"\n=== {name} ===")
    print(f"{'block':8s} {'orig var':>14s} {'orig share':>11s} {'retained var':>14s} {'retention':>10s}")
    for name_, (s, e) in blocks.items():
        loading_sq = (comps[:, s:e] ** 2).sum(axis=1)
        retained = (eigvals * loading_sq).sum()
        orig = total_var[name_]
        print(f"{name_:8s} {orig:14.2f} {orig/grand_total*100:10.1f}% "
              f"{retained:14.2f} {retained/orig*100:9.1f}%")


if __name__ == "__main__":
    analyze("checkpoints/embed_cache_mobileclip_match_72000_5000_5000", "MobileCLIP + match")
    analyze("checkpoints/embed_cache_match_72000_5000_5000", "CLIP + match")
