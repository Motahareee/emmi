import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPImageProcessor


class ImageEncoder(nn.Module):
    """
    CLIP vision encoder (ViT-B/32) — outputs 512-dim joint embeddings.

    Uses the same CLIP checkpoint as TextEncoder so both modalities are
    already in the same embedding space.  No additional alignment projection
    is required.

    Input:  pixel_values [B, 3, 224, 224]  OR  list of PIL Images
    Output: [B, 512]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 freeze_base: bool = True, **kwargs):
        super().__init__()
        self.processor       = CLIPImageProcessor.from_pretrained(model_name)
        clip                 = CLIPModel.from_pretrained(model_name)
        self.vision_model      = clip.vision_model       # CLIPVisionTransformer
        self.visual_projection = clip.visual_projection  # Linear(768 → 512)

        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images) -> torch.Tensor:
        device = next(self.vision_model.parameters()).device

        if not isinstance(images, torch.Tensor):
            inputs       = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        else:
            pixel_values = images.to(device)

        out    = self.vision_model(pixel_values=pixel_values)
        pooled = out.pooler_output              # [B, 768]
        return self.visual_projection(pooled)   # [B, 512]
