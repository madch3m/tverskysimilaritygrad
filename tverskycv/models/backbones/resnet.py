from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from registry.registry import BACKBONES


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        out_dim: int = 128,
        pretrained: bool = False,
        in_channels: int = 3,
    ):
        """
        Args:
            out_dim (int): Size of output embedding vector.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            in_channels (int): # input channels (1 for MNIST, 3 for RGB).
        """
        super().__init__()

        # Select weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        # Build base model
        self.model = resnet18(weights=weights)

        # If using grayscale input (MNIST), adapt first conv layer
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Remove classification head (fc)
        self.model.fc = nn.Identity()
        backbone_dim = 512  # resnet18 final layer before fc

        # Final projection layer
        self.proj = nn.Linear(backbone_dim, out_dim)
        self.out_dim = out_dim

        # Optional: init
        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)        # [B, 512]
        feats = self.proj(feats)     # [B, out_dim]
        return feats

    def feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.out_dim


# ---------- Registry entry ----------

@BACKBONES.register("resnet18")
def build_resnet18(out_dim: int = 128, pretrained: bool = False, in_channels: int = 3, **_):
    return ResNetBackbone(
        out_dim=out_dim,
        pretrained=pretrained,
        in_channels=in_channels,
    )