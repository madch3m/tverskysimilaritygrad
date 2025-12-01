"""
Tversky Reduce Backbone - CNN with Tversky Projection Layers

This module provides Tversky projection layers with optional GlobalFeature sharing,
and a backbone that combines CNN feature extraction with Tversky projections.

Author: Based on Tversky Neural Networks paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

# Import GlobalFeature from shared_tversky
from .shared_tversky import GlobalFeature


# ============================================================================
# Base Tversky Classes (without GlobalFeature sharing)
# ============================================================================

class TverskyCompact(nn.Module):
    """
    Compact Tversky Projection Layer - Optimized for parameter efficiency.

    Uses direct weight matrix parameterization for minimum parameter count.
    Best for deployment and production use where memory/compute is critical.

    Tversky Similarity:
        s(x, p_k) = |x ∩ p_k| / (|x ∩ p_k| + α|x \\ p_k| + β|p_k \\ x|)

    Args:
        in_features: Input dimension (e.g., 36 for MNIST conv output)
        n_prototypes: Number of output prototypes (e.g., 10 for 10 classes)
        alpha: Weight for features in input but not in prototype (default: 1.0)
        beta: Weight for features in prototype but not in input (default: 1.0)

    Example:
        >>> layer = TverskyCompact(in_features=36, n_prototypes=10)
        >>> x = torch.randn(32, 36)  # batch_size=32
        >>> output = layer(x)  # shape: (32, 10)
        >>> print(f"Parameters: {sum(p.numel() for p in layer.parameters())}")
        Parameters: 370
    """

    def __init__(
        self,
        in_features: int,
        n_prototypes: int,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()

        self.in_features = in_features
        self.n_prototypes = n_prototypes
        self.alpha = alpha
        self.beta = beta

        # Direct weight matrix: [n_prototypes × in_features]
        # This is the most compact representation
        self.weight = nn.Parameter(torch.randn(n_prototypes, in_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Kaiming (He) initialization."""
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

        # Tversky similarity formula
        # Add small epsilon for numerical stability
        denominator = intersection + self.alpha * x_diff + self.beta * w_diff + 1e-8
        similarity = intersection / denominator

        return similarity + self.bias

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (f'in_features={self.in_features}, '
                f'n_prototypes={self.n_prototypes}, '
                f'alpha={self.alpha}, beta={self.beta}')

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class TverskyInterpretable(nn.Module):
    """
    Interpretable Tversky Projection Layer - Factorized for visualization.

    Uses separate prototype and feature matrices that can be visualized
    and analyzed. Best for research, analysis, and understanding what
    the network has learned.

    Architecture:
        Input → Feature Transform (shared) → Feature Space →
        Compare to Prototypes → Similarity Scores

    The factorization enables:
        - Visualization of learned prototypes (one per class)
        - Analysis of learned features (shared representation)
        - Understanding of similarity computations

    Args:
        in_features: Input dimension (e.g., 36 for MNIST conv output)
        n_prototypes: Number of output prototypes (e.g., 10 for 10 classes)
        n_features: Intermediate feature dimension (e.g., 20)
        alpha: Weight for features in input but not in prototype (default: 1.0)
        beta: Weight for features in prototype but not in input (default: 1.0)

    Example:
        >>> layer = TverskyInterpretable(in_features=36, n_prototypes=10, n_features=20)
        >>> x = torch.randn(32, 36)
        >>> output = layer(x)  # shape: (32, 10)
        >>> prototypes, features = layer.get_learned_components()
        >>> print(f"Prototypes shape: {prototypes.shape}")  # (10, 20)
        >>> print(f"Features shape: {features.shape}")      # (20, 36)
        >>> print(f"Parameters: {layer.get_num_parameters()}")
        Parameters: 930
    """

    def __init__(
        self,
        in_features: int,
        n_prototypes: int,
        n_features: int,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()

        self.in_features = in_features
        self.n_prototypes = n_prototypes
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta

        # Factorized representation for interpretability:
        # Features: [n_features × in_features] - SHARED by all prototypes
        # Prototypes: [n_prototypes × n_features] - One per class
        self.features = nn.Parameter(torch.randn(n_features, in_features))
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Kaiming (He) initialization."""
        nn.init.kaiming_normal_(self.features, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.prototypes, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between input and prototypes.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Step 1: Project input through shared feature transformation
        # x: (batch, in_features) -> x_features: (batch, n_features)
        x_features = F.linear(x, self.features)

        # Step 2: Expand for broadcasting with prototypes
        # x_features: (batch, n_features) -> (batch, 1, n_features)
        # prototypes: (n_prototypes, n_features) -> (1, n_prototypes, n_features)
        x_expanded = x_features.unsqueeze(1)
        p_expanded = self.prototypes.unsqueeze(0)

        # Step 3: Apply sigmoid for set-theoretic interpretation
        x_sig = torch.sigmoid(x_expanded)
        p_sig = torch.sigmoid(p_expanded)

        # Step 4: Compute Tversky similarity components
        # Intersection: features present in both
        intersection = torch.min(x_sig, p_sig).sum(dim=-1)

        # Asymmetric differences
        x_diff = torch.clamp(x_sig - p_sig, min=0).sum(dim=-1)
        p_diff = torch.clamp(p_sig - x_sig, min=0).sum(dim=-1)

        # Step 5: Tversky similarity formula
        denominator = intersection + self.alpha * x_diff + self.beta * p_diff + 1e-8
        similarity = intersection / denominator

        return similarity + self.bias

    def get_learned_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the learned prototypes and features for visualization.

        Returns:
            prototypes: Tensor of shape (n_prototypes, n_features)
            features: Tensor of shape (n_features, in_features)
        """
        return self.prototypes.detach(), self.features.detach()

    def visualize_prototypes(
        self,
        class_names: Optional[list] = None,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize the learned prototype and feature matrices.

        Args:
            class_names: Optional list of class names for labeling
            figsize: Figure size for matplotlib
            save_path: If provided, save figure to this path
        """
        prototypes, features = self.get_learned_components()
        prototypes = prototypes.cpu().numpy()
        features = features.cpu().numpy()

        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.n_prototypes)]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

        # Plot prototypes
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(prototypes, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax1.set_xlabel('Feature Dimension', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Prototype (Class)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Learned Prototypes ({self.n_prototypes} classes × {self.n_features} features)',
                      fontsize=13, fontweight='bold')
        ax1.set_yticks(range(self.n_prototypes))
        ax1.set_yticklabels(class_names)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Plot features
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(features, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax2.set_xlabel('Input Dimension', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax2.set_title(f'Learned Feature Transformation ({self.n_features} features × {self.in_features} inputs)',
                      fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def get_prototype_for_class(self, class_idx: int) -> torch.Tensor:
        """
        Get the prototype vector for a specific class.

        Args:
            class_idx: Index of the class (0 to n_prototypes-1)

        Returns:
            Prototype vector of shape (n_features,)
        """
        return self.prototypes[class_idx].detach()

    def get_feature_importance(self) -> torch.Tensor:
        """
        Compute feature importance based on magnitude across all prototypes.

        Returns:
            Feature importance scores of shape (n_features,)
        """
        # Compute L2 norm of each feature across prototypes
        importance = torch.norm(self.prototypes.detach(), p=2, dim=0)
        return importance

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return (f'in_features={self.in_features}, '
                f'n_prototypes={self.n_prototypes}, '
                f'n_features={self.n_features}, '
                f'alpha={self.alpha}, beta={self.beta}')

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Shared Tversky Classes (with GlobalFeature sharing support)
# ============================================================================

class SharedTverskyCompact(TverskyCompact):
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
        # Initialize base class without alpha/beta (we'll handle them separately)
        # We need to manually set up the base class attributes
        nn.Module.__init__(self)
        self.in_features = in_features
        self.n_prototypes = n_prototypes
        
        # Weight matrix (prototypes) - always layer-specific
        self.weight = nn.Parameter(torch.randn(n_prototypes, in_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))
        
        self.feature_key = feature_key
        self.share_params = share_params
        
        # Get GlobalFeature singleton
        self._global_feature = GlobalFeature()
        
        # Tversky parameters: shared via GlobalFeature if share_params=True
        if share_params:
            param_key = f"tversky_params_{feature_key}"
            if not self._global_feature.has_key(param_key):
                params = {
                    'alpha': nn.Parameter(torch.tensor(alpha)),
                    'beta': nn.Parameter(torch.tensor(beta))
                }
                self._global_feature.register_feature(param_key, params)
                # CRITICAL: Register parameters with the module so optimizer can find them
                self.register_parameter('alpha_shared', params['alpha'])
                self.register_parameter('beta_shared', params['beta'])
            else:
                # If already exists, still register them with this module
                existing_params = self._global_feature.get_feature(param_key)
                self.register_parameter('alpha_shared', existing_params['alpha'])
                self.register_parameter('beta_shared', existing_params['beta'])
            self._param_key = param_key
            self._alpha = None
            self._beta = None
        else:
            self._alpha = nn.Parameter(torch.tensor(alpha))
            self._beta = nn.Parameter(torch.tensor(beta))
            self._param_key = None
    
    @property
    def alpha(self):
        """Get alpha parameter."""
        if self._param_key:
            params = self._global_feature.get_feature(self._param_key)
            if params and 'alpha' in params:
                return params['alpha']
            # Fallback to registered parameter if GlobalFeature access fails
            if hasattr(self, 'alpha_shared'):
                return self.alpha_shared
        return self._alpha
    
    @property
    def beta(self):
        """Get beta parameter."""
        if self._param_key:
            params = self._global_feature.get_feature(self._param_key)
            if params and 'beta' in params:
                return params['beta']
            # Fallback to registered parameter if GlobalFeature access fails
            if hasattr(self, 'beta_shared'):
                return self.beta_shared
        return self._beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky similarity between input and prototypes.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Similarity scores of shape (batch_size, n_prototypes)
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)
        w_expanded = self.weight.unsqueeze(0)
        
        # Apply sigmoid to get values in [0, 1] for set interpretation
        x_sig = torch.sigmoid(x_expanded)
        w_sig = torch.sigmoid(w_expanded)
        
        # Compute Tversky similarity components
        intersection = torch.min(x_sig, w_sig).sum(dim=-1)
        x_diff = torch.clamp(x_sig - w_sig, min=0).sum(dim=-1)
        w_diff = torch.clamp(w_sig - x_sig, min=0).sum(dim=-1)
        
        # Get alpha and beta (using properties)
        alpha = self.alpha
        beta = self.beta
        
        # Tversky similarity formula
        denominator = intersection + alpha * x_diff + beta * w_diff + 1e-8
        similarity = intersection / denominator
        
        return similarity + self.bias
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        params = sum(p.numel() for p in [self.weight, self.bias])
        if not self.share_params:
            params += self._alpha.numel() + self._beta.numel()
        return params


class SharedTverskyInterpretable(TverskyInterpretable):
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
        # Initialize base class without alpha/beta (we'll handle them separately)
        # We need to manually set up the base class attributes
        nn.Module.__init__(self)
        self.in_features = in_features
        self.n_prototypes = n_prototypes
        self.n_features = n_features
        
        # Prototypes are always layer-specific
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_features))
        self.bias = nn.Parameter(torch.zeros(n_prototypes))
        
        self.feature_key = feature_key
        self.share_features = share_features
        
        # Get GlobalFeature singleton
        self._global_feature = GlobalFeature()
        
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
                # CRITICAL: Register parameters with the module so optimizer can find them
                self.register_parameter('alpha_shared', params['alpha'])
                self.register_parameter('beta_shared', params['beta'])
            else:
                # If already exists, still register them with this module
                existing_params = self._global_feature.get_feature(param_key)
                self.register_parameter('alpha_shared', existing_params['alpha'])
                self.register_parameter('beta_shared', existing_params['beta'])
            self._param_key = param_key
            self._alpha = None
            self._beta = None
        else:
            self._alpha = nn.Parameter(torch.tensor(alpha))
            self._beta = nn.Parameter(torch.tensor(beta))
            self._param_key = None
        
        # Re-initialize parameters
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
            params = self._global_feature.get_feature(self._param_key)
            if params and 'alpha' in params:
                return params['alpha']
            # Fallback to registered parameter if GlobalFeature access fails
            if hasattr(self, 'alpha_shared'):
                return self.alpha_shared
        return self._alpha
    
    @property
    def beta_param(self):
        """Get beta parameter."""
        if self._param_key:
            params = self._global_feature.get_feature(self._param_key)
            if params and 'beta' in params:
                return params['beta']
            # Fallback to registered parameter if GlobalFeature access fails
            if hasattr(self, 'beta_shared'):
                return self.beta_shared
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


# ============================================================================
# Backbone Class
# ============================================================================

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


# ============================================================================
# Utility Functions
# ============================================================================

def compare_tversky_variants():
    """
    Compare the compact and interpretable Tversky variants.

    Demonstrates parameter counts, forward pass, and output equivalence.
    """
    print("\n" + "="*80)
    print("TVERSKY LAYER COMPARISON")
    print("="*80)

    # Configuration
    in_features = 36
    n_prototypes = 10
    n_features = 20
    batch_size = 32

    # Create both variants
    compact = TverskyCompact(in_features=in_features, n_prototypes=n_prototypes)
    interpretable = TverskyInterpretable(
        in_features=in_features,
        n_prototypes=n_prototypes,
        n_features=n_features
    )

    print(f"\n{'Variant':<20s} {'Parameters':<15s} {'Memory (KB)':<15s}")
    print("-" * 80)

    for name, model in [("Compact", compact), ("Interpretable", interpretable)]:
        params = model.get_num_parameters()
        memory_kb = params * 4 / 1024  # float32 = 4 bytes
        print(f"{name:<20s} {params:<15,} {memory_kb:<14.2f}")

    ratio = interpretable.get_num_parameters() / compact.get_num_parameters()
    print(f"\nInterpretable overhead: {ratio:.2f}x")

    # Test forward pass
    print("\n" + "="*80)
    print("FORWARD PASS TEST")
    print("="*80)

    x = torch.randn(batch_size, in_features)

    out_compact = compact(x)
    out_interpretable = interpretable(x)

    print(f"\nInput shape:              {tuple(x.shape)}")
    print(f"Compact output shape:     {tuple(out_compact.shape)}")
    print(f"Interpretable output:     {tuple(out_interpretable.shape)}")

    print("\nSample outputs (first 3 samples, first 5 classes):")
    print(f"Compact:\n{out_compact[:3, :5]}")
    print(f"\nInterpretable:\n{out_interpretable[:3, :5]}")

    # Show interpretable components
    print("\n" + "="*80)
    print("INTERPRETABLE COMPONENTS")
    print("="*80)

    prototypes, features = interpretable.get_learned_components()
    print(f"\nPrototype matrix shape:   {tuple(prototypes.shape)}")
    print(f"Feature matrix shape:     {tuple(features.shape)}")

    importance = interpretable.get_feature_importance()
    print(f"\nFeature importance (top 5 features):")
    top_features = torch.argsort(importance, descending=True)[:5]
    for i, feat_idx in enumerate(top_features):
        print(f"  {i+1}. Feature {feat_idx.item()}: {importance[feat_idx].item():.4f}")

    print("\n" + "="*80)
    print("✓ Both variants working correctly!")
    print("="*80 + "\n")


# ============================================================================
# Registry entries
# ============================================================================

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
