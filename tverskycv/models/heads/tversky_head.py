import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Prototypes Π (num_classes x feature_bank_size) - in feature space
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_bank_size) * 0.02)
        # Feature bank Ω (feature_bank_size x in_dim) - projects input to feature space
        self.features   = nn.Parameter(torch.randn(feature_bank_size, in_dim) * 0.02)

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        self.theta = nn.Parameter(torch.tensor(theta))
        self.intersection = intersection
        self.difference   = difference
        self._out = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between input and prototypes.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Project input through feature bank (shared feature transformation)
        # x: (batch, in_dim) -> x_features: (batch, feature_bank_size)
        x_features = F.linear(x, self.features)
        
        # Expand for broadcasting with prototypes
        # x_features: (batch, feature_bank_size) -> (batch, 1, feature_bank_size)
        # prototypes: (num_classes, feature_bank_size) -> (1, num_classes, feature_bank_size)
        x_expanded = x_features.unsqueeze(1)
        p_expanded = self.prototypes.unsqueeze(0)
        
        # Apply sigmoid for set-theoretic interpretation
        x_sig = torch.sigmoid(x_expanded)
        p_sig = torch.sigmoid(p_expanded)
        
        # Compute Tversky similarity components
        # Intersection: features present in both
        intersection = torch.min(x_sig, p_sig).sum(dim=-1)
        
        # Asymmetric differences
        x_diff = torch.clamp(x_sig - p_sig, min=0).sum(dim=-1)
        p_diff = torch.clamp(p_sig - x_sig, min=0).sum(dim=-1)
        
        # Tversky similarity formula
        # Add small epsilon for numerical stability
        denominator = intersection + self.alpha * x_diff + self.beta * p_diff + 1e-8
        similarity = intersection / denominator
        
        # Apply theta scaling if needed
        if self.theta != 1.0:
            similarity = similarity * self.theta
        
        return similarity

    def output_dim(self) -> int: return self._out