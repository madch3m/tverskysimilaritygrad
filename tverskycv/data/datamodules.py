import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
from typing import Optional

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


class Fruits360Dataset(Dataset):
    """PyTorch Dataset wrapper for Hugging Face fruits-360 dataset."""
    
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset: Hugging Face dataset split (e.g., ds['train'])
            transform: Optional torchvision transform
        """
        self.data = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed, then apply transform
        if isinstance(image, Image.Image):
            pass  # Already PIL Image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, list):
            # Handle list of values (flattened image)
            image_array = np.array(image, dtype=np.uint8)
            if len(image_array.shape) == 1:
                # Assume 3 channels, 64x64 (or use sqrt to determine)
                size = int(np.sqrt(len(image_array) // 3))
                image_array = image_array.reshape(3, size, size).transpose(1, 2, 0)
            image = Image.fromarray(image_array)
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy then PIL
            if image.dtype != torch.uint8:
                image = (image * 255).clamp(0, 255).to(torch.uint8)
            image_np = image.numpy()
            if len(image_np.shape) == 3 and image_np.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                image_np = image_np.transpose(1, 2, 0)
            image = Image.fromarray(image_np)
        else:
            # Try to convert to numpy then PIL
            try:
                image = Image.fromarray(np.array(image))
            except:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transform (resize, to_tensor, normalize)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label


class Fruits360DataModule:
    """Data module for fruits-360 dataset from Hugging Face."""
    
    def __init__(
        self,
        dataset_name: str = "PedroSampaio/fruits-360",
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 64,
        in_channels: int = 3,
        **_
    ):
        """
        Args:
            dataset_name: Hugging Face dataset name
            data_dir: Local cache directory for dataset
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            img_size: Target image size (assumed square)
            in_channels: Number of input channels (3 for RGB)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for Fruits360DataModule. "
                "Install it with: pip install datasets"
            )
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Load dataset from Hugging Face
        print(f"Loading {dataset_name} from Hugging Face...")
        self.ds = load_dataset(dataset_name, cache_dir=data_dir)
        
        # Apply transformations
        print("Applying transformations...")
        self._transform_dataset()
        
        # Create PyTorch datasets
        self.train = Fruits360Dataset(self.train_data_transformed, transform=self.transform)
        
        # Use test split if available, otherwise use validation
        if 'test' in self.ds:
            test_cols_to_remove = [col for col in self.ds['test'].column_names if col not in ['image', 'label']]
            self.val_data_transformed = self.ds['test'].map(
                self._transform_batch,
                batched=True,
                batch_size=100,
                remove_columns=test_cols_to_remove if test_cols_to_remove else None
            )
        elif 'validation' in self.ds:
            val_cols_to_remove = [col for col in self.ds['validation'].column_names if col not in ['image', 'label']]
            self.val_data_transformed = self.ds['validation'].map(
                self._transform_batch,
                batched=True,
                batch_size=100,
                remove_columns=val_cols_to_remove if val_cols_to_remove else None
            )
        else:
            # Fallback: use a subset of train for validation
            print("Warning: No test/validation split found. Using train split for validation.")
            self.val_data_transformed = self.train_data_transformed
        
        self.val = Fruits360Dataset(self.val_data_transformed, transform=self.transform)
        
        print(f"✓ Dataset loaded: {len(self.train)} train samples, {len(self.val)} val samples")
    
    def _transform_batch(self, examples):
        """Transform a batch of images - resize only, full transform in __getitem__."""
        images = examples['image']
        transformed_images = []
        
        resize_transform = transforms.Resize((self.img_size, self.img_size))
        
        for img in images:
            if isinstance(img, Image.Image):
                img_resized = resize_transform(img)
            elif isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_resized = resize_transform(img_pil)
            else:
                # For other types, try to convert
                try:
                    img_pil = Image.fromarray(np.array(img))
                    img_resized = resize_transform(img_pil)
                except:
                    img_resized = img
            
            # Store as PIL Image (will be converted to tensor in __getitem__)
            transformed_images.append(img_resized)
        
        examples['image'] = transformed_images
        return examples
    
    def _transform_dataset(self):
        """Apply transformations to the dataset."""
        train_cols_to_remove = [col for col in self.ds['train'].column_names if col not in ['image', 'label']]
        self.train_data_transformed = self.ds['train'].map(
            self._transform_batch,
            batched=True,
            batch_size=100,
            remove_columns=train_cols_to_remove if train_cols_to_remove else None
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
>>>>>>> 1f0c45d (Adding early checkpoint and yaml files to work with onnix script)
