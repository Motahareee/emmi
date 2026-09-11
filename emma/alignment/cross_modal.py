import torch
import torch.nn as nn


FUSION_DIM_FACTOR = {"mean": 1, "concat": 2, "match": 4}


class CrossModalAlignment(nn.Module):
    """
    Fuses CLIP text and image embeddings into a single representation.

    Both encoders share the same joint embedding space, so no cross-modal
    projection is needed.  Three parameter-free fusion modes (parameter-free
    so fused embeddings stay cacheable with fully frozen encoders):

      mean   : (t + v) / 2                  → [B, d]     (default)
      concat : [t ; v]                      → [B, 2d]    keeps both modalities
      match  : [t ; v ; |t - v| ; t * v]    → [B, 4d]    pair-classification
               features — difference/product terms directly encode agreement

    Input:  text  [B, d], image [B, d]
    Output: [B, d * FUSION_DIM_FACTOR[fusion]]
    """

    def __init__(self, d_shared: int = 512, fusion: str = "mean", **kwargs):
        super().__init__()
        if fusion not in FUSION_DIM_FACTOR:
            raise ValueError(f"unknown fusion mode: {fusion!r}")
        self.d_shared = d_shared
        self.fusion   = fusion
        self.d_out    = d_shared * FUSION_DIM_FACTOR[fusion]

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if self.fusion == "mean":
            return (text + image) / 2
        if self.fusion == "concat":
            return torch.cat([text, image], dim=-1)
        # match
        return torch.cat([text, image,
                          (text - image).abs(),
                          text * image], dim=-1)
