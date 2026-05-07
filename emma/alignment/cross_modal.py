import torch
import torch.nn as nn


class CrossModalAlignment(nn.Module):
    """
    Projects text and audio embeddings into a shared space and fuses them.

    Two fusion modes:
      - mean      : element-wise average of the two projected embeddings
      - attention : self-attention over the two modality tokens, then mean pool

    Vision is not included — Facet 4.2 features are unavailable in the
    current setup and will be added as a future extension.

    Input:  text  [B, d_text]
            audio [B, d_audio]
    Output: [B, d_shared]
    """

    def __init__(self, d_text: int, d_audio: int,
                 d_shared: int = 256, use_attention_fusion: bool = False,
                 # kept for API compatibility — ignored
                 d_vision: int = 0):
        super().__init__()

        self.proj_text  = nn.Sequential(nn.Linear(d_text,  d_shared), nn.ReLU())
        self.proj_audio = nn.Sequential(nn.Linear(d_audio, d_shared), nn.ReLU())

        self.use_attention = use_attention_fusion
        if use_attention_fusion:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_shared, num_heads=4, batch_first=True
            )
            self.norm = nn.LayerNorm(d_shared)

    def forward(self, text: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor = None) -> torch.Tensor:
        t = self.proj_text(text)    # [B, d_shared]
        a = self.proj_audio(audio)  # [B, d_shared]

        tokens = torch.stack([t, a], dim=1)   # [B, 2, d_shared]

        if self.use_attention:
            fused, _ = self.attn(tokens, tokens, tokens)
            fused = self.norm(fused + tokens)
            return fused.mean(dim=1)           # [B, d_shared]

        return tokens.mean(dim=1)              # [B, d_shared]
