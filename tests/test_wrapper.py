import torch
import torch.nn as nn
import torch.optim as optim

from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.models.backbones.simple_cnn import SimpleCNN
from tverskycv.models.heads.linear_head import LinearHead
from tverskycv.registry import BACKBONES, HEADS


def test_wrapper_forward_shape():
    torch.manual_seed(0)
    backbone = SimpleCNN(out_dim=32)
    head = LinearHead(in_dim=32, num_classes=10)
    model = ImageClassifier(backbone, head)

    x = torch.randn(4, 1, 28, 28)  # MNIST-like batch
    logits = model(x)
    assert logits.shape == (4, 10)


def test_wrapper_grad_and_step_reduces_loss():
    torch.manual_seed(0)
    backbone = SimpleCNN(out_dim=64)
    head = LinearHead(in_dim=64, num_classes=10)
    model = ImageClassifier(backbone, head)

    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # small synthetic batch
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))

    # measure initial loss
    model.train()
    logits = model(x)
    loss0 = criterion(logits, y).item()

    # take a few steps
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    loss1 = loss.item()

    # loss should decrease (allow tiny numerical wiggle)
    assert loss1 <= loss0 + 1e-6


def test_wrapper_with_registry_builders():
    torch.manual_seed(0)

    # Build via registry to ensure wiring is correct
    backbone = BACKBONES.get("simple_cnn")(out_dim=48)
    head = HEADS.get("linear")(in_dim=48, num_classes=7)
    model = ImageClassifier(backbone, head)

    x = torch.randn(5, 1, 28, 28)
    y = model(x)
    assert y.shape == (5, 7)


def test_wrapper_moves_to_device_cpu():
    torch.manual_seed(0)
    model = ImageClassifier(SimpleCNN(out_dim=16), LinearHead(in_dim=16, num_classes=3))
    model = model.to("cpu")
    x = torch.randn(2, 1, 28, 28, device="cpu")
    y = model(x)
    assert y.device.type == "cpu"
    assert y.shape == (2, 3)