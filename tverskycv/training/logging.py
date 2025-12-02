#!/usr/bin/env python3
"""
Unified logging utilities for training.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import torch.nn as nn

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None


class TrainingLogger:
    """
    Comprehensive logging for training progress.
    Supports both file logging and TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: Path | str,
        use_tensorboard: bool = True,
        log_file: str = 'training.log'
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to enable TensorBoard logging
            log_file: Name of the text log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        else:
            self.writer = None
            if use_tensorboard and not TENSORBOARD_AVAILABLE:
                print("⚠ TensorBoard not available, logging to file only")
        
        self.log_file_path = self.log_dir / log_file
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def log(self, message: str, print_console: bool = True):
        """
        Log message to file and optionally console.
        
        Args:
            message: Message to log
            print_console: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Learning rate
        """
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['learning_rate'].append(lr)
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        acc: float,
        lr: Optional[float] = None
    ):
        """
        Log batch-level metrics.
        
        Args:
            epoch: Epoch number
            batch_idx: Batch index
            total_batches: Total number of batches
            loss: Batch loss
            acc: Batch accuracy
            lr: Optional learning rate
        """
        if self.use_tensorboard and self.writer is not None:
            global_step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss, global_step)
            self.writer.add_scalar('Accuracy/Train_Batch', acc, global_step)
            if lr is not None:
                self.writer.add_scalar('Learning_Rate_Batch', lr, global_step)
    
    def log_model_weights(self, model: nn.Module, epoch: int):
        """
        Log model weight histograms to TensorBoard.
        
        Args:
            model: PyTorch model
            epoch: Epoch number
        """
        if self.use_tensorboard and self.writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Weights/{name}', param.cpu(), epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad.cpu(), epoch)
    
    def save_metrics(self, filepath: str = 'metrics.json'):
        """
        Save metrics history to JSON file.
        
        Args:
            filepath: Name of the metrics file
        """
        filepath = self.log_dir / filepath
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to {filepath}")
    
    def close(self):
        """Close the logger and flush all buffers."""
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        self.log("Training logger closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

