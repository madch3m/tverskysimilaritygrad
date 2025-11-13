import torch
import torch.nn as nn
import torch.nn.functional as F
from .shared_tversky import TverskyTransformerBlock, GlobalFeature
from transformers import GPT2Config


class TverskyAttentionBackbone(nn.Module):
    """
    Vision Transformer-like backbone using Tversky Attention for image classification.
    Converts images to patches, projects to embeddings, and applies Tversky Transformer blocks.
    """
    def __init__(
        self,
        out_dim: int = 128,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_key: str = 'main',
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: convert image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Class token (optional, for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Create GPT2-like config for transformer blocks
        config = GPT2Config(
            n_embed=embed_dim,
            n_head=num_heads,
            n_inner=embed_dim * 4,
            resid_pdrop=dropout,
            layer_norm_epsilon=1e-5,
        )
        
        # Tversky Transformer blocks
        self.blocks = nn.ModuleList([
            TverskyTransformerBlock(
                config,
                feature_key=f"{feature_key}_layer_{i}",
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            for i in range(num_layers)
        ])
        
        # Layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, out_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, out_dim)
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.patch_embed(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten spatial dimensions: (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            outputs = block(x, use_cache=False, output_attentions=False)
            x = outputs[0]
        
        # Layer norm
        x = self.ln_f(x)
        
        # Use class token for classification
        x = x[:, 0]  # (B, embed_dim)
        
        # Project to output dimension
        x = self.proj(x)  # (B, out_dim)
        
        return x

