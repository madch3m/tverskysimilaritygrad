from ...registry.registry import HEADS
from .linear_head import LinearHead
from .tversky_head import TverskyProjectionHead

@HEADS.register("linear")
def build_linear(in_dim: int, num_classes: int, **_): 
    return LinearHead(in_dim, num_classes)

@HEADS.register("tversky")
def build_tversky(in_dim: int, num_classes: int, feature_bank_size: int=256, **kw):
    return TverskyProjectionHead(in_dim, num_classes, feature_bank_size=feature_bank_size, **kw)