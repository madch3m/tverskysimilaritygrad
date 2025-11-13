import torch
import torch.nn as nn

from tverskycv.registry import BACKBONES
from tverskycv.models.backbones.simple_cnn import SimpleCNN

# Try importing resnet18 builder; if torchvision is missing in CI, we can skip resnet tests gracefully.
_RESNET_AVAILABLE = True
try:
    from tverskycv.models.backbones.resnet import build_resnet18
except Exception:
    _RESNET_AVAILABLE = False

# Try importing tversky_attention builder; if transformers is missing in CI, we can skip tversky tests gracefully.
_TVERSKY_ATTENTION_AVAILABLE = True
try:
    from tverskycv.models.backbones.tversky_attention import TverskyAttentionBackbone
except Exception:
    _TVERSKY_ATTENTION_AVAILABLE = False


def test_simple_cnn_forward_shape():
    torch.manual_seed(0)
    model = SimpleCNN(out_dim=128)
    x = torch.randn(4, 1, 28, 28)  # MNIST-like
    y = model(x)
    assert y.shape == (4, 128)


def test_simple_cnn_grad_flow():
    torch.manual_seed(0)
    model = SimpleCNN(out_dim=64)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)  # (2, 64)
    target = torch.zeros_like(out)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    # Ensure at least one parameter received gradients
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)












def test_registry_has_backbones():
    # registry should have keys for the backbones we registered
    assert BACKBONES.get("simple_cnn") is not None
    # resnet18 may not be present if torchvision import failed; handle both cases
    try:
        fn = BACKBONES.get("resnet18")
        # get() should raise KeyError if missing; if it doesn't, ensure callable
        assert callable(fn)
    except KeyError:
        # acceptable if resnet18 wasn't registered in this environment
        pass
    # tversky_attention may not be present if transformers import failed; handle both cases
    try:
        fn = BACKBONES.get("tversky_attention")
        assert callable(fn)
    except KeyError:
        # acceptable if tversky_attention wasn't registered in this environment
        pass
