from ...registry.registry import BACKBONES
from .simple_cnn import SimpleCNN
from .tversky_attention import TverskyAttentionBackbone
from .tversky_gpt import create_tversky_gpt_from_config, TverskyGPTModel, count_parameters, get_shared_parameter_info
from .tversky_reduce_backbone import (
    TverskyReduceBackbone,
    SharedTverskyCompact,
    SharedTverskyInterpretable
)

@BACKBONES.register("simple_cnn")
def build_simple_cnn(out_dim: int = 128, **_): return SimpleCNN(out_dim=out_dim)

@BACKBONES.register("tversky_attention")
def build_tversky_attention(
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
    **_
):
    return TverskyAttentionBackbone(
        out_dim=out_dim,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        feature_key=feature_key,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )