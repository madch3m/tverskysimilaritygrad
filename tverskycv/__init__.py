# Import modules that register things (side-effect imports)
from . import data  # triggers DATASETS registration
from .models import backbones, heads  # triggers BACKBONES/HEADS registration

__all__ = ["data", "models"]