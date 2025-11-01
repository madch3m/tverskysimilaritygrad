import torch.nn as nn
from .base import IProjectionHead

class LinearHead(IProjectionHead):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self._out = num_classes

    def forward(self, x): return self.fc(x)
    def output_dim(self): return self._out