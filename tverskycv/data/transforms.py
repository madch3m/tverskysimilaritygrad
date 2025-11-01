# tverskycv/data/transforms.py
"""
Transform utilities for datasets.

Currently minimal — MNIST requires only ToTensor(),
but this file provides a consistent interface so new
datasets (e.g., CIFAR-10) can add augmentation later.
"""

from typing import Tuple
from torchvision import transforms


def build_transforms(
    train: bool = True,
    normalize: bool = False,
) -> transforms.Compose:
    t = []

    t.append(transforms.ToTensor())

    # ---- Data augmentation (none for MNIST; add later) ----
    if train:
        # You can enable future augmentation here, e.g.
        #
        # t.append(transforms.RandomRotation(10))
        # t.append(transforms.RandomCrop(28, padding=2))
        #
        pass

    # ---- Optional normalization ----
    if normalize:
        # Standard MNIST stats — OK to use; safe no-op if disabled
        t.append(transforms.Normalize((0.1307,), (0.3081,)))

    return transforms.Compose(t)


def default_train_val_transforms(
    normalize: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Convenience helper returning both train + val transforms.
    """
    return (
        build_transforms(train=True, normalize=normalize),
        build_transforms(train=False, normalize=normalize),
    )