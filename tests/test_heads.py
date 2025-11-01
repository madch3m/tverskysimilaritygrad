import torch
import torch.nn as nn
import pytest

from tverskycv.registry import HEADS

# Direct imports for concrete classes (safer than going only through registry in tests)
from tverskycv.models.heads.linear_head import LinearHead

# Tversky head may have forward() unimplemented initially; test should reflect that reality.
try:
    from tverskycv.models.heads.tversky_head import TverskyProjectionHead
    _TVERSKY_AVAILABLE = True
except Exception:
    _TVERSKY_AVAILABLE = False


def test_linear_head_forward_shape_and_output_dim():
    torch.manual_seed(0)
    in_dim, num_classes = 128, 10
    head = LinearHead(in_dim=in_dim, num_classes=num_classes)

    x = torch.randn(4, in_dim)  # features from backbone
    logits = head(x)

    assert logits.shape == (4, num_classes)
    assert head.output_dim() == num_classes


def test_linear_head_grad_flow():
    torch.manual_seed(0)
    in_dim, num_classes = 64, 7
    head = LinearHead(in_dim=in_dim, num_classes=num_classes)

    x = torch.randn(2, in_dim)
    target = torch.randint(0, num_classes, (2,))

    loss = nn.CrossEntropyLoss()(head(x), target)
    loss.backward()

    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_registry_has_linear_and_can_build():
    builder = HEADS.get("linear")
    head = builder(in_dim=32, num_classes=5)
    x = torch.randn(3, 32)
    y = head(x)
    assert y.shape == (3, 5)


@pytest.mark.skipif(not _TVERSKY_AVAILABLE, reason="Tversky head module not available")
def test_registry_has_tversky_and_constructor_works():
    builder = HEADS.get("tversky")
    head = builder(in_dim=16, num_classes=3, feature_bank_size=8)
    assert head.output_dim() == 3


@pytest.mark.skipif(not _TVERSKY_AVAILABLE, reason="Tversky head module not available")
def test_tversky_forward_not_implemented_yet():
    # Until the training team implements forward(), calling it should raise NotImplementedError.
    head = TverskyProjectionHead(in_dim=8, num_classes=2, feature_bank_size=4)
    x = torch.randn(2, 8)
    with pytest.raises(NotImplementedError):
        _ = head(x)