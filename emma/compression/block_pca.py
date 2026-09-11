"""
Block-PCA compression for match-fused embeddings.

Applies PCA independently to each structural block of the match-fused vector:
    x = [t ; v ; |t-v| ; t⊙v]   shape [B, 4 * d_block]

Each 512-dim block gets n_per_block PCA components (default: 16).
Total output: 4 * n_per_block = 64 dims.

Key insight: |t-v| and t⊙v capture the discriminative signal between modalities.
These blocks are genuinely high-variance across match/mismatch pairs, so PCA on
each block finds that signal — unlike global PCA on mean fusion which destroys it.

Only meaningful with match fusion (fusion='match'), which produces the 4-block input.
"""

import torch
import numpy as np
from pathlib import Path

from emma.compression.pca import PCACompressor


class BlockPCACompressor:
    """
    PCA applied per structural block of a match-fused vector.

    encode : project each block to n_per_block PCA components → [B, 4*n_per_block]
    decode : reconstruct each block from its components       → [B, 4*d_block]
    """

    def __init__(self, n_components: int = 64, d_block: int = 512):
        assert n_components % 4 == 0, "n_components must be divisible by 4"
        self.n_components  = n_components
        self.d_block       = d_block
        self.n_per_block   = n_components // 4
        self._pcas = [PCACompressor(n_components=self.n_per_block) for _ in range(4)]

    def _split_blocks(self, x: torch.Tensor):
        """Split [B, 4*d_block] into four [B, d_block] tensors."""
        d = self.d_block
        return x[:, :d], x[:, d:2*d], x[:, 2*d:3*d], x[:, 3*d:]

    def fit(self, X: torch.Tensor):
        """
        Fit one PCACompressor per block.
        Args:
            X : [N, 4*d_block]  match-fused embeddings from training split
        """
        blocks = self._split_blocks(X)
        names  = ["t", "v", "|t-v|", "t⊙v"]
        for i, (pca, block, name) in enumerate(zip(self._pcas, blocks, names)):
            print(f"  Block {i} ({name}): ", end="")
            pca.fit(block)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4*d_block] → [B, n_components]"""
        blocks = self._split_blocks(x)
        parts  = [pca.encode(block) for pca, block in zip(self._pcas, blocks)]
        return torch.cat(parts, dim=1)   # [B, 4*n_per_block]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, n_components] → [B, 4*d_block]"""
        n = self.n_per_block
        parts = [z[:, i*n:(i+1)*n] for i in range(4)]
        recon = [pca.decode(part) for pca, part in zip(self._pcas, parts)]
        return torch.cat(recon, dim=1)   # [B, 4*d_block]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        for i, pca in enumerate(self._pcas):
            pca.save(f"{path}/block_{i}")
        np.save(f"{path}/meta.npy",
                np.array([self.n_components, self.d_block]))

    @classmethod
    def load(cls, path: str) -> "BlockPCACompressor":
        meta = np.load(f"{path}/meta.npy")
        obj  = cls(n_components=int(meta[0]), d_block=int(meta[1]))
        obj._pcas = [PCACompressor.load(f"{path}/block_{i}") for i in range(4)]
        return obj
