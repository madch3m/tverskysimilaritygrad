import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from tverskycv.registry.registry import DATASETS

class MNISTDataModule:
    def __init__(self, data_dir="./data", batch_size=32, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.ToTensor()

        self.train = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.val = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
