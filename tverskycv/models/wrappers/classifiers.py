import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)   # (B, D)
        logits = self.head(feats)  # (B, C)
        return logits