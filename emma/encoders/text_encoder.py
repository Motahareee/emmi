import torch
import torch.nn as nn
from transformers import CLIPModel


class TextEncoder(nn.Module):
    """
    CLIP text encoder (ViT-B/32) — outputs 512-dim joint embeddings.

    Uses the same CLIP checkpoint as ImageEncoder so both modalities are
    already in the same embedding space.  No additional alignment projection
    is required.

    Input:  input_ids [B, L], attention_mask [B, L]  (CLIPTokenizer, L ≤ 77)
    Output: [B, 512]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 freeze_base: bool = True, **kwargs):
        super().__init__()
        clip = CLIPModel.from_pretrained(model_name)
        self.text_model      = clip.text_model        # CLIPTextTransformer
        self.text_projection = clip.text_projection   # Linear(512 → 512)

        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out    = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output           # [B, 512]  EOS-token embedding
        return self.text_projection(pooled)  # [B, 512]  joint CLIP space
