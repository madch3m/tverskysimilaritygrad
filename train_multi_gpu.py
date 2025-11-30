#!/usr/bin/env python3

"""

Standalone script for multi-GPU training on cloud platforms.

Usage: python train_multi_gpu.py --gpus 4 --batch-size 256 --epochs 10

"""



import argparse

import torch

from datasets import load_dataset

from torchvision import transforms

from PIL import Image

import numpy as np

from torch.utils.data import Dataset



# Import Tversky modules

from tverskycv.models.backbones.shared_tversky import SharedTverskyLinear, GlobalFeature

from tverskycv.training import launch_distributed_training

import torch.nn as nn

import torch.nn.functional as F





class FruitsClassifierWithSharing(nn.Module):

    """Fruit classifier using SharedTverskyLinear with feature sharing."""

    

    def __init__(self, input_dim, hidden_dims, num_classes, feature_key='fruits', share_features=True):

        super().__init__()

        

        self.layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [num_classes]

        

        for i in range(len(dims) - 1):

            layer = SharedTverskyLinear(

                in_features=dims[i],

                out_features=dims[i + 1],

                feature_key=feature_key,

                alpha=0.5,

                beta=0.5,

                gamma=1.0,

                share_features=share_features

            )

            self.layers.append(layer)

    

    def forward(self, x):

        for i, layer in enumerate(self.layers[:-1]):

            x = layer(x)

            x = F.relu(x)

        x = self.layers[-1](x)

        return x





class Fruits360Dataset(Dataset):

    """PyTorch Dataset wrapper for Fruits-360."""

    

    def __init__(self, data):

        self.data = data

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        item = self.data[idx]

        image = item['image']

        label = item['label']

        

        if not isinstance(image, torch.Tensor):

            if isinstance(image, list):

                image = torch.tensor(image)

            elif isinstance(image, np.ndarray):

                image = torch.from_numpy(image)

            else:

                image = torch.tensor(image)

        

        return image.flatten(), label





def load_and_prepare_data(img_size=64):

    """Load and prepare Fruits-360 dataset."""

    print("Loading fruits-360 dataset...")

    ds = load_dataset("PedroSampaio/fruits-360")

    

    transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    

    def transform_dataset(examples):

        images = examples['image']

        transformed_images = []

        

        for img in images:

            if isinstance(img, Image.Image):

                img_tensor = transform(img)

            elif isinstance(img, np.ndarray):

                img_pil = Image.fromarray(img)

                img_tensor = transform(img_pil)

            else:

                img_tensor = torch.tensor(img) if not isinstance(img, torch.Tensor) else img

            transformed_images.append(img_tensor.numpy())

        

        examples['image'] = transformed_images

        return examples

    

    print("Transforming dataset...")

    train_data = ds['train'].map(

        transform_dataset,

        batched=True,

        batch_size=100,

        remove_columns=[col for col in ds['train'].column_names if col not in ['image', 'label']]

    )

    

    test_data = ds['test' if 'test' in ds else 'validation'].map(

        transform_dataset,

        batched=True,

        batch_size=100,

        remove_columns=[col for col in ds['test' if 'test' in ds else 'validation'].column_names if col not in ['image', 'label']]

    )

    

    train_dataset = Fruits360Dataset(train_data)

    test_dataset = Fruits360Dataset(test_data)

    

    num_classes = len(set(ds['train']['label']))

    feature_dim = img_size * img_size * 3

    

    return train_dataset, test_dataset, num_classes, feature_dim





def main():

    parser = argparse.ArgumentParser(description='Multi-GPU training for Tversky Fruits-360')

    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use')

    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per GPU')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--img-size', type=int, default=64, help='Image size')

    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256, 128], help='Hidden dimensions')

    

    args = parser.parse_args()

    

    # Load data

    train_dataset, test_dataset, num_classes, feature_dim = load_and_prepare_data(args.img_size)

    

    print(f"\nDataset loaded:")

    print(f"  Train samples: {len(train_dataset):,}")

    print(f"  Test samples: {len(test_dataset):,}")

    print(f"  Classes: {num_classes}")

    print(f"  Feature dim: {feature_dim}")

    

    # Model creation function

    def create_model():

        gf = GlobalFeature()

        gf.clear()

        

        return FruitsClassifierWithSharing(

            input_dim=feature_dim,

            hidden_dims=args.hidden_dims,

            num_classes=num_classes,

            feature_key='fruits_shared',

            share_features=True

        )

    

    # Launch training

    print(f"\nLaunching training on {args.gpus} GPUs...")

    print(f"  Batch size per GPU: {args.batch_size}")

    print(f"  Total batch size: {args.batch_size * args.gpus}")

    print(f"  Epochs: {args.epochs}")

    print(f"  Learning rate: {args.lr}")

    

    launch_distributed_training(

        model_fn=create_model,

        train_dataset=train_dataset,

        test_dataset=test_dataset,

        world_size=args.gpus,

        num_epochs=args.epochs,

        batch_size=args.batch_size,

        learning_rate=args.lr

    )

    

    print("\n✓ Training complete!")





if __name__ == '__main__':

    main()

