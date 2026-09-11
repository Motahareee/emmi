"""
PCA-based linear compression — baseline for comparison with VAE.

Fits a PCA on training fused embeddings, then encodes/decodes any split.
Saved as a plain dict so it has no torch dependency at load time.

Interface mirrors VAE:
    compressor.encode(x)  → [B, d_latent]  (numpy or tensor in, tensor out)
    compressor.decode(z)  → [B, d_in]
"""

import torch
import numpy as np
from pathlib import Path


class PCACompressor:
    """
    Linear encoder/decoder via PCA.

    encode : project to top-k principal components  → [B, d_latent]
    decode : reconstruct from principal components  → [B, d_in]
    """

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.mean_    = None   # [d_in]
        self.components_ = None  # [n_components, d_in]

    def fit(self, X: torch.Tensor):
        """
        Fit PCA on training embeddings.
        Args:
            X : [N, d_in]  fused embeddings from the full training split
        """
        X_np = X.numpy().astype(np.float32)
        self.mean_ = X_np.mean(axis=0)
        X_c = X_np - self.mean_
        # SVD — more numerically stable than covariance eigendecomposition
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]   # [n_components, d_in]
        var_total    = (X_c ** 2).sum()
        var_captured = ((X_c @ self.components_.T) ** 2).sum()
        print(f"  PCA fitted: {self.n_components} components, "
              f"variance explained: {var_captured / var_total:.3f}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, d_in] → [B, n_components]"""
        x_np = x.cpu().numpy().astype(np.float32)
        z    = (x_np - self.mean_) @ self.components_.T   # [B, n_components]
        return torch.tensor(z, dtype=torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, n_components] → [B, d_in]"""
        z_np = z.cpu().numpy().astype(np.float32)
        x    = z_np @ self.components_ + self.mean_       # [B, d_in]
        return torch.tensor(x, dtype=torch.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 mean=self.mean_,
                 components=self.components_,
                 n_components=np.array([self.n_components]))

    @classmethod
    def load(cls, path: str) -> "PCACompressor":
        data = np.load(path + ".npz")
        obj  = cls(n_components=int(data["n_components"][0]))
        obj.mean_       = data["mean"]
        obj.components_ = data["components"]
        return obj
