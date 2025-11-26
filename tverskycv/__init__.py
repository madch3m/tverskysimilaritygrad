"""
TverskyCV - Tversky Similarity-based Computer Vision Framework
"""
# Import key modules to ensure registrations are executed
# Use try/except to handle optional dependencies gracefully
try:
    from . import models  # noqa: F401
except ImportError:
    pass  # Some models may require optional dependencies

try:
    from . import data  # noqa: F401
except ImportError:
    pass  # Some datasets may require optional dependencies

try:
    from . import training  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
