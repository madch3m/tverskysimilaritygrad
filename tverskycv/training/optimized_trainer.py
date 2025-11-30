#!/usr/bin/env python3

"""

Optimized training utilities for Tversky CV models.

Supports single-GPU and multi-GPU (DDP) training with mixed precision.

"""



import torch

import torch.nn as nn

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

import time
import os
from typing import Optional, Dict, Any

# Conditional import for mixed precision (requires CUDA)
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    # Fallback for CPU-only systems
    AMP_AVAILABLE = False
    # Create dummy context manager for autocast
    class autocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    # Create dummy GradScaler
    class GradScaler:
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass





class OptimizedTrainer:

    """

    Optimized trainer for Tversky models with support for:

    - Mixed precision training (AMP)

    - Automatic batch size selection

    - Learning rate scheduling

    - Gradient clipping

    - Best model checkpointing

    """

    

    def __init__(

        self,

        model: nn.Module,

        device: torch.device,

        num_epochs: int = 10,

        learning_rate: float = 0.001,

        weight_decay: float = 0.01,

        label_smoothing: float = 0.1,

        gradient_clip: float = 1.0,

        checkpoint_dir: str = 'checkpoints'

    ):

        self.model = model.to(device)

        self.device = device

        self.num_epochs = num_epochs

        self.checkpoint_dir = checkpoint_dir

        self.gradient_clip = gradient_clip

        

        # Create checkpoint directory

        os.makedirs(checkpoint_dir, exist_ok=True)

        

        # Loss function with label smoothing

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        

        # Optimizer

        self.optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay,

            betas=(0.9, 0.999)

        )

        

        # Mixed precision scaler (only if AMP available)
        if AMP_AVAILABLE and device.type == 'cuda':
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
            if device.type == 'cpu':
                print("⚠ Mixed precision (AMP) not available on CPU - using FP32")

        

        # Metrics tracking

        self.best_val_acc = 0.0

        self.train_losses = []

        self.train_accs = []

        self.val_losses = []

        self.val_accs = []

        self.epoch_times = []

        

    def setup_scheduler(self, steps_per_epoch: int):

        """Setup OneCycleLR scheduler."""

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(

            self.optimizer,

            max_lr=0.01,

            epochs=self.num_epochs,

            steps_per_epoch=steps_per_epoch,

            pct_start=0.3,

            anneal_strategy='cos'

        )

    

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:

        """Train for one epoch."""

        self.model.train()

        epoch_loss = 0.0

        correct = 0

        total = 0

        

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(self.device, non_blocking=True)

            labels = labels.to(self.device, non_blocking=True)

            

            self.optimizer.zero_grad(set_to_none=True)

            

            # Mixed precision forward pass (if available)
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                
                # Optimizer step
                self.optimizer.step()
            
            self.scheduler.step()

            

            # Statistics

            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            

            # Progress update

            if (batch_idx + 1) % 100 == 0:

                current_acc = correct / total

                print(f"  Epoch {epoch+1}/{self.num_epochs} - Batch {batch_idx+1}/{len(train_loader)}: "

                      f"Loss={loss.item():.4f}, Acc={current_acc:.4f}, "

                      f"LR={self.scheduler.get_last_lr()[0]:.6f}")

        

        return {

            'loss': epoch_loss / len(train_loader),

            'accuracy': correct / total

        }

    

    @torch.no_grad()

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:

        """Validate the model."""

        self.model.eval()

        epoch_loss = 0.0

        correct = 0

        total = 0

        

        for images, labels in val_loader:

            images = images.to(self.device, non_blocking=True)

            labels = labels.to(self.device, non_blocking=True)

            

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            

            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

        

        return {

            'loss': epoch_loss / len(val_loader),

            'accuracy': correct / total

        }

    

    def save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, is_best: bool = False):

        """Save model checkpoint."""

        checkpoint = {

            'epoch': epoch,

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'scheduler_state_dict': self.scheduler.state_dict(),

            'val_acc': val_acc,

            'val_loss': val_loss,

            'best_val_acc': self.best_val_acc,

        }

        

        # Save regular checkpoint

        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)

        

        # Save best model

        if is_best:

            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

            torch.save(checkpoint, best_path)

            print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f}")

    

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:

        """Complete training loop."""

        # Setup scheduler

        self.setup_scheduler(len(train_loader))

        

        # Enable optimizations

        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():

            torch.backends.cuda.matmul.allow_tf32 = True

            torch.backends.cudnn.allow_tf32 = True

        

        print(f"\n{'='*70}")

        print(f"Starting Training")

        print(f"{'='*70}")

        print(f"Device: {self.device}")

        if torch.cuda.is_available():

            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"Epochs: {self.num_epochs}")

        print(f"Batches per epoch: {len(train_loader)}")

        print(f"{'='*70}\n")

        

        total_start = time.time()

        

        for epoch in range(self.num_epochs):

            epoch_start = time.time()

            

            # Train

            train_metrics = self.train_epoch(train_loader, epoch)

            

            # Validate

            val_metrics = self.validate(val_loader)

            

            # Track metrics

            epoch_time = time.time() - epoch_start

            self.epoch_times.append(epoch_time)

            self.train_losses.append(train_metrics['loss'])

            self.train_accs.append(train_metrics['accuracy'])

            self.val_losses.append(val_metrics['loss'])

            self.val_accs.append(val_metrics['accuracy'])

            

            # Print summary

            print(f"\n{'='*70}")

            print(f"Epoch {epoch+1}/{self.num_epochs} Summary - Time: {epoch_time:.2f}s")

            print(f"{'='*70}")

            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            

            # Save checkpoint

            is_best = val_metrics['accuracy'] > self.best_val_acc

            if is_best:

                self.best_val_acc = val_metrics['accuracy']

            

            self.save_checkpoint(epoch, val_metrics['accuracy'], val_metrics['loss'], is_best)

            print(f"{'='*70}\n")

        

        # Final summary

        total_time = time.time() - total_start

        

        print(f"\n{'='*70}")

        print(f"Training Complete!")

        print(f"{'='*70}")

        print(f"Total time: {total_time/60:.2f} minutes")

        print(f"Average time per epoch: {sum(self.epoch_times)/len(self.epoch_times):.2f} seconds")

        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        print(f"{'='*70}\n")

        

        return {

            'train_losses': self.train_losses,

            'train_accs': self.train_accs,

            'val_losses': self.val_losses,

            'val_accs': self.val_accs,

            'best_val_acc': self.best_val_acc,

            'total_time': total_time,

            'epoch_times': self.epoch_times

        }





class DistributedTrainer:

    """

    Distributed Data Parallel (DDP) trainer for multi-GPU training.

    """

    

    def __init__(

        self,

        rank: int,

        world_size: int,

        model: nn.Module,

        num_epochs: int = 10,

        learning_rate: float = 0.001,

        weight_decay: float = 0.01,

        checkpoint_dir: str = 'checkpoints'

    ):

        self.rank = rank

        self.world_size = world_size

        self.num_epochs = num_epochs

        self.checkpoint_dir = checkpoint_dir

        

        # Setup distributed training

        self.setup_distributed()

        

        # Move model to GPU

        torch.cuda.set_device(rank)

        self.model = model.to(rank)

        self.model = DDP(model, device_ids=[rank])

        

        # Loss and optimizer

        self.criterion = nn.CrossEntropyLoss().to(rank)

        self.optimizer = torch.optim.AdamW(

            self.model.parameters(),

            lr=learning_rate,

            weight_decay=weight_decay

        )

        

        # Mixed precision (only if available)
        if AMP_AVAILABLE:
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False

        

        # Create checkpoint dir (only on rank 0)

        if rank == 0:

            os.makedirs(checkpoint_dir, exist_ok=True)

    

    def setup_distributed(self):

        """Initialize distributed training."""

        os.environ['MASTER_ADDR'] = 'localhost'

        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    

    def cleanup(self):

        """Cleanup distributed training."""

        dist.destroy_process_group()

    

    def train(self, train_dataset, val_dataset, batch_size: int = 128):

        """Train with DDP."""

        # Create distributed samplers

        train_sampler = DistributedSampler(

            train_dataset,

            num_replicas=self.world_size,

            rank=self.rank,

            shuffle=True

        )

        

        val_sampler = DistributedSampler(

            val_dataset,

            num_replicas=self.world_size,

            rank=self.rank,

            shuffle=False

        )

        

        # Create data loaders

        train_loader = DataLoader(

            train_dataset,

            batch_size=batch_size,

            sampler=train_sampler,

            num_workers=4,

            pin_memory=True,

            persistent_workers=True

        )

        

        val_loader = DataLoader(

            val_dataset,

            batch_size=batch_size,

            sampler=val_sampler,

            num_workers=4,

            pin_memory=True,

            persistent_workers=True

        )

        

        # Setup scheduler

        scheduler = torch.optim.lr_scheduler.OneCycleLR(

            self.optimizer,

            max_lr=0.01,

            epochs=self.num_epochs,

            steps_per_epoch=len(train_loader)

        )

        

        # Training loop

        for epoch in range(self.num_epochs):

            train_sampler.set_epoch(epoch)

            

            # Train

            self.model.train()

            for batch_idx, (images, labels) in enumerate(train_loader):

                images = images.to(self.rank, non_blocking=True)

                labels = labels.to(self.rank, non_blocking=True)

                

                self.optimizer.zero_grad(set_to_none=True)

                

                if self.use_amp:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                scheduler.step()

                

                if self.rank == 0 and (batch_idx + 1) % 50 == 0:

                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            

            # Validate

            self.model.eval()

            val_correct = 0

            val_total = 0

            

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(self.rank, non_blocking=True)

                    labels = labels.to(self.rank, non_blocking=True)

                    

                    if self.use_amp:
                        with autocast():
                            logits = self.model(images)
                    else:
                        logits = self.model(images)

                    

                    preds = torch.argmax(logits, dim=-1)

                    val_correct += (preds == labels).sum().item()

                    val_total += labels.size(0)

            

            val_acc = val_correct / val_total

            

            # Aggregate across GPUs

            val_acc_tensor = torch.tensor([val_acc]).to(self.rank)

            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)

            

            if self.rank == 0:

                print(f"\nEpoch {epoch+1}/{self.num_epochs}: Val Acc: {val_acc_tensor.item():.4f}\n")

                

                # Save checkpoint

                if (epoch + 1) % 5 == 0:

                    torch.save({

                        'epoch': epoch,

                        'model_state_dict': self.model.module.state_dict(),

                        'val_acc': val_acc_tensor.item(),

                    }, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}_ddp.pth'))

        

        self.cleanup()





def get_optimal_batch_size(gpu_name: str) -> int:

    """Get optimal batch size based on GPU memory."""

    gpu_configs = {

        'T4': 128,

        'V100': 256,

        'A100': 512,

        'RTX 6000': 512,

        'RTX 4090': 256,

    }

    for key in gpu_configs:

        if key in gpu_name:

            return gpu_configs[key]

    return 128





def create_optimized_dataloaders(

    train_dataset,

    test_dataset,

    batch_size: Optional[int] = None,

    num_workers: int = 4

) -> tuple:

    """Create optimized data loaders."""

    

    # Auto-detect optimal batch size if not provided

    if batch_size is None and torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        batch_size = get_optimal_batch_size(gpu_name)

        print(f"Auto-detected batch size: {batch_size} for {gpu_name}")

    elif batch_size is None:

        batch_size = 32

    

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=num_workers > 0,

        prefetch_factor=2 if num_workers > 0 else None,

        drop_last=True

    )

    

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=num_workers > 0

    )

    

    return train_loader, test_loader

