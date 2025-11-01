import torch
import torch.nn as nn
from .base import IProjectionHead

class TverskyProjectionHead(IProjectionHead):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        feature_bank_size: int = 256,
        alpha: float = 0.5,
        beta: float = 0.5,
        theta: float = 1.0,
        intersection: str = "product",   # "min","max","mean","softmin","gmean"
        difference: str = "subtractmatch" # or "ignorematch"
    ):
        super().__init__()
        # Prototypes Π (num_classes x in_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        # Feature bank Ω (feature_bank_size x in_dim)
        self.features   = nn.Parameter(torch.randn(feature_bank_size, in_dim) * 0.02)

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        self.theta = nn.Parameter(torch.tensor(theta))
        self.intersection = intersection
        self.difference   = difference
        self._out = num_classes

    #implement forward to compute Tversky similarity logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim). Compute SΩ,α,β,θ(x, Πj) for all classes j.
        # NOTE: Leave function bodies for the team to flesh out using the paper’s eqns.
        # Return logits (B, num_classes).
        raise NotImplementedError("Implement differentiable Tversky similarity here.")

    def output_dim(self) -> int: return self._out