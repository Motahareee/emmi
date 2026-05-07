"""
Lightweight projectors for CMU-MOSEI pre-extracted features.

These replace the full encoders (DistilBERT, CNN+GRU, MobileNetV3) when
working with the SDK's pre-computed features instead of raw data.
They share the same output interface: given a feature tensor they return
a single [B, d_out] embedding, so the rest of the pipeline is unchanged.

Input shapes accepted:
    [B, T, d_in]  — sequence features (temporal mean-pooling applied)
    [B, d_in]     — already pooled features
"""

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """
    Linear projection from pre-extracted feature space to d_out.

    Args:
        d_in  : input feature dimension (300 text / 74 audio / 35 vision)
        d_out : output embedding size — must match the corresponding
                d_text / d_audio / d_vision in EncoderConfig
        mask  : if True, uses the boolean mask for weighted mean pooling
                instead of a simple mean (respects padding)
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features : [B, T, d_in] or [B, d_in]
            mask     : [B, T] bool, True = valid position (optional)
        Returns:
            [B, d_out]
        """
        if features.dim() == 3:
            if mask is not None:
                # Masked mean: ignore padding positions
                m = mask.unsqueeze(-1).float()         # [B, T, 1]
                features = (features * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                features = features.mean(dim=1)        # [B, d_in]

        return self.proj(features)                     # [B, d_out]


def build_mosei_projectors(d_text: int, d_audio: int,
                            d_vision: int) -> dict:
    """
    Convenience factory — returns a dict with the three projectors
    pre-configured for CMU-MOSEI feature dimensions.

    Usage
    -----
    from emma.encoders.feature_projector import build_mosei_projectors
    from emma.data import FEATURE_DIMS

    proj = build_mosei_projectors(d_text=256, d_audio=256, d_vision=256)
    edge = EdgePipeline(cfg,
                        text_encoder=proj["text"],
                        audio_encoder=proj["audio"],
                        vision_encoder=proj["vision"])
    """
    from emma.data.mosei import FEATURE_DIMS
    return {
        "text":   ModalityProjector(FEATURE_DIMS["text"],   d_text),
        "audio":  ModalityProjector(FEATURE_DIMS["audio"],  d_audio),
        "vision": ModalityProjector(FEATURE_DIMS["vision"], d_vision),
    }
