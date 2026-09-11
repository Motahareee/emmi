"""
LDA-PCA hybrid compressor for task-aware dimensionality reduction.

Standard PCA finds directions of maximum variance, which may not align with
the task-relevant signal (as observed with MobileCLIP embeddings).

This compressor uses Fisher's Linear Discriminant Analysis to find the single
most class-discriminative direction first, then fills the remaining components
with PCA directions in the orthogonal complement.  For binary classification
LDA yields exactly 1 discriminant — the component that best separates matched
from non-matched pairs.  The remaining n_components-1 directions preserve as
much reconstruction fidelity as possible while being orthogonal to the LDA
direction.

Interface mirrors PCACompressor:
    compressor.fit(X, y)   — requires labels (binary 0/1)
    compressor.encode(x)   → [B, n_components]
    compressor.decode(z)   → [B, d_in]
    compressor.save(path)
    LDACompressor.load(path)
"""

import torch
import numpy as np
from pathlib import Path


class LDACompressor:
    """
    LDA direction (1st component) + PCA complement (remaining components).

    encode : project onto [lda_direction | pca_complement]  → [B, n_components]
    decode : approximate reconstruction via pseudo-inverse   → [B, d_in]
    """

    def __init__(self, n_components: int = 64):
        self.n_components  = n_components
        self.mean_         = None   # [d_in]  global mean (for centering)
        self.components_   = None   # [n_components, d_in]  orthonormal rows

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit LDA-PCA on training embeddings and binary labels.

        Args:
            X : [N, d_in]  fused embeddings
            y : [N]        binary labels (0 / 1)
        """
        X_np = X.numpy().astype(np.float64)
        y_np = y.numpy().astype(np.int32)

        self.mean_ = X_np.mean(axis=0).astype(np.float32)
        X_c = X_np - self.mean_

        # ── Step 1: Fisher's LDA discriminant direction ───────────────────────
        mask0 = (y_np == 0)
        mask1 = (y_np == 1)
        mu0   = X_c[mask0].mean(axis=0)
        mu1   = X_c[mask1].mean(axis=0)

        # Within-class scatter (regularised for numerical stability)
        S0 = (X_c[mask0] - mu0).T @ (X_c[mask0] - mu0)
        S1 = (X_c[mask1] - mu1).T @ (X_c[mask1] - mu1)
        Sw = S0 + S1
        eps = 1e-6 * np.trace(Sw) / Sw.shape[0]
        Sw += eps * np.eye(Sw.shape[0])

        # w = Sw^{-1} (mu1 - mu0)
        diff  = (mu1 - mu0).reshape(-1, 1)
        w_lda = np.linalg.solve(Sw, diff).ravel()
        w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-12)   # unit norm
        w_lda = w_lda.astype(np.float32)

        print(f"  LDA: class means separation = "
              f"{float(np.abs(w_lda @ (mu1 - mu0))):.4f}")

        if self.n_components == 1:
            self.components_ = w_lda[None, :]
            return

        # ── Step 2: Project out LDA direction, run PCA on residual ───────────
        X_c32     = X_c.astype(np.float32)
        proj_lda  = (X_c32 @ w_lda)[:, None] * w_lda[None, :]   # [N, d_in]
        X_resid   = X_c32 - proj_lda                              # [N, d_in]

        n_pca = self.n_components - 1
        _, _, Vt = np.linalg.svd(X_resid, full_matrices=False)
        pca_comp = Vt[:n_pca].astype(np.float32)                  # [n_pca, d_in]

        # variance explained by PCA complement
        var_total    = float((X_resid ** 2).sum())
        var_captured = float(((X_resid @ pca_comp.T) ** 2).sum())
        print(f"  PCA complement: {n_pca} components, "
              f"variance in residual explained: {var_captured/var_total:.3f}")

        # ── Stack: [lda_direction; pca_complement] — rows are orthonormal ────
        self.components_ = np.vstack([w_lda[None, :], pca_comp])  # [n_components, d_in]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, d_in] → [B, n_components]"""
        x_np = x.cpu().numpy().astype(np.float32)
        z    = (x_np - self.mean_) @ self.components_.T
        return torch.tensor(z, dtype=torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, n_components] → [B, d_in]  (approximate reconstruction)"""
        z_np = z.cpu().numpy().astype(np.float32)
        x    = z_np @ self.components_ + self.mean_
        return torch.tensor(x, dtype=torch.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 mean=self.mean_,
                 components=self.components_,
                 n_components=np.array([self.n_components]))

    @classmethod
    def load(cls, path: str) -> "LDACompressor":
        data = np.load(path + ".npz")
        obj  = cls(n_components=int(data["n_components"][0]))
        obj.mean_       = data["mean"]
        obj.components_ = data["components"]
        return obj
