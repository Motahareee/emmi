import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


class VisionEncoder(nn.Module):
    """
    MobileNetV3-Small backbone over K sampled keyframes, averaged across frames.

    Input:  frames [B, K, C, H, W]
    Output: [B, d_out]

    MobileNetV3-Small classifier layout:
        Linear(576, 1024) → Hardswish → Dropout → Linear(1024, num_classes)
    We replace the final Linear with a projection to d_out.
    """

    def __init__(self, d_out: int = 256, freeze_base: bool = True,
                 pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Replace the final classification layer with a projection
        in_features = backbone.classifier[-1].in_features    # 1024
        backbone.classifier[-1] = nn.Linear(in_features, d_out)
        self.backbone = backbone

        if freeze_base:
            # Freeze everything except the new projection layer
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        self.norm = nn.LayerNorm(d_out)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, K, C, H, W]
        B, K, C, H, W = frames.shape
        flat = frames.view(B * K, C, H, W)        # [B*K, C, H, W]
        encoded = self.backbone(flat)              # [B*K, d_out]
        encoded = encoded.view(B, K, -1)           # [B, K, d_out]
        pooled = encoded.mean(dim=1)               # [B, d_out] — avg over keyframes
        return self.norm(pooled)
