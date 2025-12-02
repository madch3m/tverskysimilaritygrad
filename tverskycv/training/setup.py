#!/usr/bin/env python3
"""
Training setup utilities for unified configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from argparse import Namespace

from .utils import resolve_device
from .optimizers import build_optimizer
from .schedulers import build_scheduler


def setup_training_from_config(
    cfg: Dict[str, Any],
    model: nn.Module,
    args: Optional[Namespace] = None
) -> Dict[str, Any]:
    """
    Set up all training components from configuration.
    
    Args:
        cfg: Configuration dictionary
        model: PyTorch model
        args: Optional CLI arguments to override config
        
    Returns:
        Dictionary containing:
            - optimizer: Optimizer instance
            - scheduler: Scheduler instance (or None)
            - criterion: Loss function
            - device: torch.device
            - checkpoint_dir: Path for checkpoints
            - epochs: Number of training epochs
            - learning_rate: Initial learning rate
    """
    # Resolve device
    device_str = None
    if args and hasattr(args, 'device') and args.device is not None:
        device_str = args.device
    elif 'train' in cfg and 'device' in cfg['train']:
        device_str = cfg['train']['device']
    
    device = resolve_device(device_str)
    
    # Build optimizer
    train_cfg = cfg.get('train', {})
    optimizer_cfg = cfg.get('optimizer', {})
    
    # Default optimizer config if not provided
    if not optimizer_cfg:
        lr = train_cfg.get('lr', 0.001)
        optimizer_cfg = {
            'name': 'adamw',
            'params': {
                'lr': lr,
                'weight_decay': train_cfg.get('weight_decay', 0.01)
            }
        }
    elif 'params' not in optimizer_cfg:
        # Ensure lr is in params
        if 'lr' not in optimizer_cfg.get('params', {}):
            optimizer_cfg['params'] = optimizer_cfg.get('params', {})
            optimizer_cfg['params']['lr'] = train_cfg.get('lr', 0.001)
    
    optimizer = build_optimizer(model.parameters(), optimizer_cfg)
    
    # Build scheduler
    scheduler_cfg = train_cfg.get('scheduler', None)
    scheduler = build_scheduler(optimizer, scheduler_cfg)
    
    # Create criterion
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    if label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Get checkpoint directory
    checkpoint_dir = train_cfg.get('ckpt_dir', './checkpoints')
    
    # Get epochs
    epochs = train_cfg.get('epochs', 10)
    
    # Get learning rate (from optimizer)
    learning_rate = optimizer.param_groups[0]['lr']
    
    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'device': device,
        'checkpoint_dir': checkpoint_dir,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }

