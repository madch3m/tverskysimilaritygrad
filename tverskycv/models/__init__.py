"""
Model components: backbones, heads, and wrappers.
"""
# Import backbones to trigger registration decorators
# Note: Some backbones may require optional dependencies (e.g., transformers)
try:
    from . import backbones  # noqa: F401
except ImportError as e:
    # Optional dependencies may be missing, but core functionality should still work
    import warnings
    warnings.warn(f"Some backbones could not be loaded: {e}")

# Import heads to trigger registration decorators
try:
    from . import heads  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"Some heads could not be loaded: {e}")

# Import wrappers
from .wrappers.classifiers import ImageClassifier  # noqa: F401

__all__ = [
    "ImageClassifier",
]

