import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    DistilBERT backbone with a linear projection head.

    Input:  input_ids [B, L], attention_mask [B, L]
    Output: [B, d_out]
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 d_out: int = 256, freeze_base: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        d_model = self.backbone.config.hidden_size       # 768 for DistilBERT

        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # [B, d_model]
        return self.proj(pooled)                             # [B, d_out]
