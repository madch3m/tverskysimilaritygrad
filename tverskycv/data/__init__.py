from ..registry.registry import DATASETS
from .datamodules import MNISTDataModule

@DATASETS.register("mnist")
def build_mnist(**kw): return MNISTDataModule(**kw)