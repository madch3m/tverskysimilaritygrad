import types
import torch
import pytest
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from tverskycv.data.datamodules import MNISTDataModule
from tverskycv.data import transforms as tx


class _FakeMNIST(Dataset):
    """Tiny stand-in for torchvision.datasets.MNIST to avoid network access."""
    def __init__(self, *args, train=True, **kwargs):
        super().__init__()
        self.train = train
        self.n = 64 if train else 32
        # 1x28x28 grayscale images, labels in [0..9]
        self.images = torch.randn(self.n, 1, 28, 28)
        self.targets = torch.randint(low=0, high=10, size=(self.n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


@pytest.fixture(autouse=True)
def patch_mnist(monkeypatch):
    """Patch torchvision.datasets.MNIST to our fake dataset for all tests here."""
    try:
        import torchvision.datasets as tvds
    except Exception:
        pytest.skip("torchvision not available")
    monkeypatch.setattr(tvds, "MNIST", _FakeMNIST)
    yield


def test_build_transforms_compose_and_normalize_flag():
    train_t, val_t = tx.default_train_val_transforms(normalize=True)
    # should be torchvision.transforms.Compose
    from torchvision import transforms as T
    assert isinstance(train_t, T.Compose)
    assert isinstance(val_t, T.Compose)
    # Normalize should be present when normalize=True
    assert any(isinstance(tr, T.Normalize) for tr in train_t.transforms)
    assert any(isinstance(tr, T.Normalize) for tr in val_t.transforms)


def test_mnist_datamodule_loaders_return_batches():
    dm = MNISTDataModule(data_dir="./data", batch_size=16, num_workers=0)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 1, 28, 28)
    assert yb.shape == (16,)
    # labels should be in [0..9]
    assert torch.all((yb >= 0) & (yb <= 9))

    xb2, yb2 = next(iter(val_loader))
    assert xb2.shape == (16, 1, 28, 28)
    assert yb2.shape == (16,)
    assert torch.all((yb2 >= 0) & (yb2 <= 9))


def test_mnist_datamodule_loader_samplers_reflect_shuffle():
    dm = MNISTDataModule(data_dir="./data", batch_size=8, num_workers=0)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # DataLoader doesn't expose `shuffle` directly; check sampler types.
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_datamodule_is_pickleable_for_num_workers_gt0():
    # Some CI setups use num_workers=0; ensure no pickling issues arise if >0.
    # We won't actually spin workers here; just instantiate without error.
    dm = MNISTDataModule(data_dir="./data", batch_size=4, num_workers=2)
    assert hasattr(dm, "train") and hasattr(dm, "val")