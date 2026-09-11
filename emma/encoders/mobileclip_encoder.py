"""
MobileCLIP2-S0 encoders — edge-efficient drop-in replacement for CLIP ViT-B/32.

Loaded via open_clip ('MobileCLIP2-S0', pretrained='dfndr2b'):
  - image encoder: 11.4M params (vs ~87M for ViT-B/32), input 256x256
  - text encoder:  63.4M params
  - both output 512-dim embeddings in a shared joint space (same as CLIP),
    so the fusion and compression stages are unchanged.

Weights are fetched from the HuggingFace Hub — on the cluster, download once
on a login node (HF_HOME set) so offline compute nodes hit the cache.
"""

import torch
import torch.nn as nn

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"


def _load_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    return model, preprocess


class MobileCLIPTextEncoder(nn.Module):
    """
    Input:  input_ids [B, 77]  (open_clip tokenizer; attention_mask ignored —
            open_clip uses fixed-length context with argmax-EOS pooling)
    Output: [B, 512]
    """

    def __init__(self, model: nn.Module, freeze_base: bool = True):
        super().__init__()
        self.model = model
        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.model.encode_text(input_ids, normalize=False)


class MobileCLIPImageEncoder(nn.Module):
    """
    Input:  pixel_values [B, 3, 256, 256]  (open_clip preprocess)
    Output: [B, 512]
    """

    def __init__(self, model: nn.Module, freeze_base: bool = True):
        super().__init__()
        self.model = model
        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values, normalize=False)


def build_mobileclip_encoders(freeze_base: bool = True):
    """
    Build text + image encoders sharing one underlying open_clip model
    (loads weights once).  Returns (text_encoder, image_encoder).
    """
    model, _ = _load_model()
    return (MobileCLIPTextEncoder(model, freeze_base),
            MobileCLIPImageEncoder(model, freeze_base))


# ── HF-style adapters for the data pipeline ───────────────────────────────────
# COCOMatchingDataset expects a CLIPImageProcessor-like `processor` and a
# CLIPTokenizer-like `tokenizer`; these adapters give open_clip the same API.

class MobileCLIPImageProcessorAdapter:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, images, return_tensors=None):
        if not isinstance(images, (list, tuple)):
            images = [images]
        pv = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        return {"pixel_values": pv}


class MobileCLIPTokenizerAdapter:
    """open_clip tokenizer → HF-style dict. Fixed 77-token context."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, max_length=None, padding=None,
                 truncation=None, return_tensors=None):
        if isinstance(text, str):
            text = [text]
        ids = self.tokenizer(text)                       # [B, 77]
        return {"input_ids": ids,
                "attention_mask": (ids != 0).long()}


def build_mobileclip_processors():
    """Returns (image_processor, tokenizer) adapters for the data pipeline."""
    import open_clip
    _, preprocess = _load_model()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return (MobileCLIPImageProcessorAdapter(preprocess),
            MobileCLIPTokenizerAdapter(tokenizer))
