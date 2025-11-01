from ...registry.registry import BACKBONES
from .simple_cnn import SimpleCNN

@BACKBONES.register("simple_cnn")
def build_simple_cnn(out_dim: int = 128, **_): return SimpleCNN(out_dim=out_dim)