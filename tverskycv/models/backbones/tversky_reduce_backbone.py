"""
Tversky Reduce Backbone - CNN with Tversky Projection Layers

This module provides a backbone that combines CNN feature extraction with
Tversky projection layers that can share features via GlobalFeature bank.

Author: Based on Tversky Neural Networks paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import GlobalFeature from shared_tversky
from .shared_tversky import GlobalFeature


class SharedTverskyCompact(nn.Module):
    """
    Compact Tversky Projection Layer with GlobalFeature sharing support.
    
    Extends TverskyCompact to optionally share Tversky parameters (alpha, beta) 
    via GlobalFeature bank. Prototypes remain layer-specific.
    
    Args:
        in_features: Input dimension
        n_prototypes: Number of output prototypes
        feature_key: Key for GlobalFeature bank (for sharing)
        share_params: Whether to use GlobalFeature for sharing Tversky parameters
        alpha: Weight for features in input but not in prototype
        beta: Weight for features in prototype but not in input
    """
    
    def __init__(
        self,
        in_features: int,
        n_prototypes: int,
        feature_key: str = "main",
        share_params: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.n_prototypes = n_prototypes
        self.feature_key = feature_key
        self.share_params = share_params
        
        # Get GlobalFeature singleton
        self._global_feature = GlobalFeature()
        
        # Weight matrix (prototypes) - always layer-specific
        self.weight = nn.Parameter(torch.randn(n_prototypes, in_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))
        
        # Tversky parameters: shared via GlobalFeature if share_params=True
        if share_params:
            param_key = f"tversky_params_{feature_key}"
            if not self._global_feature.has_key(param_key):
                params = {
                    'alpha': nn.Parameter(torch.tensor(alpha)),
                    'beta': nn.Parameter(torch.tensor(beta))
                }
                self._global_feature.register_feature(param_key, params)
            self._param_key = param_key
            self._alpha = None
            self._beta = None
        else:
            self._alpha = nn.Parameter(torch.tensor(alpha))
            self._beta = nn.Parameter(torch.tensor(beta))
            self._param_key = None
        
        self._reset_parameters()
    
    @property
    def alpha(self):
        """Get alpha parameter."""
        if self._param_key:
            return self._global_feature.get_feature(self._param_key)['alpha']
        return self._alpha
    
    @property
    def beta(self):
        """Get beta parameter."""
        if self._param_key:
            return self._global_feature.get_feature(self._param_key)['beta']
        return self._beta
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between input and prototypes.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Expand dimensions for broadcasting
        # x: (batch, in_features) -> (batch, 1, in_features)
        # weight: (n_prototypes, in_features) -> (1, n_prototypes, in_features)
        x_expanded = x.unsqueeze(1)
        w_expanded = self.weight.unsqueeze(0)
        
        # Apply sigmoid to get values in [0, 1] for set interpretation
        x_sig = torch.sigmoid(x_expanded)
        w_sig = torch.sigmoid(w_expanded)
        
        # Compute Tversky similarity components
        # Intersection: min(x, w) represents shared features
        intersection = torch.min(x_sig, w_sig).sum(dim=-1)
        
        # x \ w: features present in x but not in w
        x_diff = torch.clamp(x_sig - w_sig, min=0).sum(dim=-1)
        
        # w \ x: features present in w but not in x
        w_diff = torch.clamp(w_sig - x_sig, min=0).sum(dim=-1)
        
        # Get alpha and beta
        alpha = self.alpha
        beta = self.beta
        
        # Tversky similarity formula
        # Add small epsilon for numerical stability
        denominator = intersection + alpha * x_diff + beta * w_diff + 1e-8
        similarity = intersection / denominator
        
        return similarity + self.bias
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        params = sum(p.numel() for p in [self.weight, self.bias])
        if not self.share_params:
            params += self._alpha.numel() + self._beta.numel()
        return params


class SharedTverskyInterpretable(nn.Module):
    """
    Interpretable Tversky Projection Layer with GlobalFeature sharing support.
    
    Extends TverskyInterpretable to optionally share feature matrices via GlobalFeature bank.
    
    Args:
        in_features: Input dimension
        n_prototypes: Number of output prototypes
        n_features: Intermediate feature dimension
        feature_key: Key for GlobalFeature bank (for sharing)
        share_features: Whether to use GlobalFeature for sharing
        alpha: Weight for features in input but not in prototype
        beta: Weight for features in prototype but not in input
    """
    
    def __init__(
        self,
        in_features: int,
        n_prototypes: int,
        n_features: int,
        feature_key: str = "main",
        share_features: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        super(SharedTverskyInterpretable, self).__init__()
        
        self.in_features = in_features
        self.n_prototypes = n_prototypes
        self.n_features = n_features
        self.feature_key = feature_key
        self.share_features = share_features
        self.alpha = alpha
        self.beta = beta
        
        # Get GlobalFeature singleton
        self._global_feature = GlobalFeature()
        
        # Prototypes are always layer-specific
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))
        
        # Feature matrix: shared via GlobalFeature if share_features=True
        if share_features:
            feature_matrix_key = f"{feature_key}_interpretable_{in_features}_{n_features}"
            if not self._global_feature.has_key(feature_matrix_key):
                features = nn.Parameter(torch.randn(n_features, in_features))
                self._global_feature.register_feature(feature_matrix_key, features)
            self._feature_matrix_key = feature_matrix_key
        else:
            self.features = nn.Parameter(torch.randn(n_features, in_features))
            self._feature_matrix_key = None
        
        # Tversky parameters: shared via GlobalFeature if share_features=True
        if share_features:
            param_key = f"tversky_params_{feature_key}"
            if not self._global_feature.has_key(param_key):
                params = {
                    'alpha': nn.Parameter(torch.tensor(alpha)),
                    'beta': nn.Parameter(torch.tensor(beta))
                }
                self._global_feature.register_feature(param_key, params)
            self._param_key = param_key
            self._alpha = None
            self._beta = None
        else:
            self._alpha = nn.Parameter(torch.tensor(alpha))
            self._beta = nn.Parameter(torch.tensor(beta))
            self._param_key = None
        
        self._reset_parameters()
    
    @property
    def features_matrix(self):
        """Get feature matrix."""
        if self.share_features and self._feature_matrix_key:
            return self._global_feature.get_feature(self._feature_matrix_key)
        return self.features
    
    @property
    def alpha_param(self):
        """Get alpha parameter."""
        if self._param_key:
            return self._global_feature.get_feature(self._param_key)['alpha']
        return self._alpha
    
    @property
    def beta_param(self):
        """Get beta parameter."""
        if self._param_key:
            return self._global_feature.get_feature(self._param_key)['beta']
        return self._beta
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.prototypes, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
        if self.share_features and self._feature_matrix_key:
            features = self._global_feature.get_feature(self._feature_matrix_key)
            nn.init.kaiming_normal_(features, mode='fan_in', nonlinearity='relu')
        elif not self.share_features:
            nn.init.kaiming_normal_(self.features, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between input and prototypes.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Step 1: Project input through shared feature transformation
        features = self.features_matrix
        x_features = F.linear(x, features)  # (batch, n_features)
        
        # Step 2: Expand for broadcasting with prototypes
        x_expanded = x_features.unsqueeze(1)  # (batch, 1, n_features)
        p_expanded = self.prototypes.unsqueeze(0)  # (1, n_prototypes, n_features)
        
        # Step 3: Apply sigmoid for set-theoretic interpretation
        x_sig = torch.sigmoid(x_expanded)
        p_sig = torch.sigmoid(p_expanded)
        
        # Step 4: Compute Tversky similarity components
        intersection = torch.min(x_sig, p_sig).sum(dim=-1)
        x_diff = torch.clamp(x_sig - p_sig, min=0).sum(dim=-1)
        p_diff = torch.clamp(p_sig - x_sig, min=0).sum(dim=-1)
        
        # Get alpha and beta
        alpha = self.alpha_param
        beta = self.beta_param
        
        # Step 5: Tversky similarity formula
        denominator = intersection + alpha * x_diff + beta * p_diff + 1e-8
        similarity = intersection / denominator
        
        return similarity + self.bias
    
    def get_learned_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the learned prototypes and features for visualization.
        
        Returns:
            prototypes: Tensor of shape (n_prototypes, n_features)
            features: Tensor of shape (n_features, in_features)
        """
        features = self.features_matrix
        return self.prototypes.detach(), features.detach()
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        params = sum(p.numel() for p in [self.prototypes, self.bias])
        if not self.share_features:
            params += self.features.numel()
        if not self._param_key:
            params += self._alpha.numel() + self._beta.numel()
        return params


class TverskyReduceBackbone(nn.Module):
    """
    CNN backbone with Tversky projection layers using GlobalFeature sharing.
    
    This backbone combines:
    - CNN feature extraction (similar to SimpleCNN)
    - Tversky projection layers (compact or interpretable)
    - GlobalFeature bank for parameter sharing
    
    Args:
        out_dim: Output feature dimension (number of prototypes)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        img_size: Input image size (assumed square)
        variant: 'compact' or 'interpretable'
        n_features: Intermediate feature dimension (for interpretable variant)
        feature_key: Key for GlobalFeature bank
        share_features: Whether to use GlobalFeature for sharing
        alpha: Tversky alpha parameter
        beta: Tversky beta parameter
    """
    
    def __init__(
        self,
        out_dim: int = 128,
        in_channels: int = 1,
        img_size: int = 28,
        variant: str = 'compact',
        n_features: int = 64,
        feature_key: str = 'main',
        share_features: bool = True,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.img_size = img_size
        self.variant = variant
        self.feature_key = feature_key
        self.share_features = share_features
        
        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
        )
        
        # Calculate flattened feature dimension
        # After two MaxPool2d(2), size is img_size // 4
        self.conv_out_dim = 64 * (img_size // 4) * (img_size // 4)
        
        # Tversky projection layer
        if variant == 'compact':
            self.tversky_proj = SharedTverskyCompact(
                in_features=self.conv_out_dim,
                n_prototypes=out_dim,
                feature_key=feature_key,
                share_params=share_features,
                alpha=alpha,
                beta=beta
            )
        elif variant == 'interpretable':
            self.tversky_proj = SharedTverskyInterpretable(
                in_features=self.conv_out_dim,
                n_prototypes=out_dim,
                n_features=n_features,
                feature_key=feature_key,
                share_features=share_features,
                alpha=alpha,
                beta=beta
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Must be 'compact' or 'interpretable'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, img_size, img_size)
        
        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        # Extract features with CNN
        features = self.feature_extractor(x)  # (batch, conv_out_dim)
        
        # Project through Tversky layer
        output = self.tversky_proj(features)  # (batch, out_dim)
        
        return output
    
    def feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.out_dim
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        conv_params = sum(p.numel() for p in self.feature_extractor.parameters())
        tversky_params = self.tversky_proj.get_num_parameters()
        return conv_params + tversky_params


# ---------- Registry entry ----------

from ...registry.registry import BACKBONES


@BACKBONES.register("tversky_reduce_compact")
def build_tversky_reduce_compact(
    out_dim: int = 128,
    in_channels: int = 1,
    img_size: int = 28,
    feature_key: str = 'main',
    share_features: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    **_
):
    """Build compact Tversky reduce backbone with GlobalFeature sharing."""
    return TverskyReduceBackbone(
        out_dim=out_dim,
        in_channels=in_channels,
        img_size=img_size,
        variant='compact',
        feature_key=feature_key,
        share_features=share_features,
        alpha=alpha,
        beta=beta,
    )


@BACKBONES.register("tversky_reduce_interpretable")
def build_tversky_reduce_interpretable(
    out_dim: int = 128,
    in_channels: int = 1,
    img_size: int = 28,
    n_features: int = 64,
    feature_key: str = 'main',
    share_features: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
    **_
):
    """Build interpretable Tversky."""
    return TverskyReduceBackbone(
        out_dim=out_dim,
        in_channels=in_channels,
        img_size=img_size,
        variant='interpretable',
        n_features=n_features,
        feature_key=feature_key,
        share_features=share_features,
        alpha=alpha,
        beta=beta,
    )

