from ..registry.registry import DATASETS
from .datamodules import MNISTDataModule, Fruits360DataModule

@DATASETS.register("mnist")
def build_mnist(**kw): return MNISTDataModule(**kw)

@DATASETS.register("fruits_360")
def build_fruits_360(**kw): return Fruits360DataModule(**kw)