# add other data modules as needed
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNISTDataModule:
    def __init__(self, data_dir: str="./data", batch_size: int=128, num_workers: int=2):
        tf = transforms.Compose([transforms.ToTensor()])
        self.train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        self.val   = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        self.bs, self.num_workers = batch_size, num_workers

    def train_dataloader(self): return DataLoader(self.train, batch_size=self.bs, shuffle=True,  num_workers=self.num_workers)
    def val_dataloader(self):   return DataLoader(self.val,   batch_size=self.bs, shuffle=False, num_workers=self.num_workers)