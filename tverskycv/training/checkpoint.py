#!/usr/bin/env python3
"""
Unified checkpoint utilities for loading and saving model checkpoints.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from .utils import save_checkpoint as _save_checkpoint_base, load_checkpoint as _load_checkpoint_base


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        device: Device to load checkpoint on
        optimizer: Optional optimizer to restore state
        
    Returns:
        Checkpoint dictionary with metadata, or None if file doesn't exist
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None
    
    try:
        # Try loading with the base utility
        ckpt = _load_checkpoint_base(
            str(path),
            model,
            optimizer,
            str(device)
        )
        return ckpt
    except Exception as e:
        # Fallback: try loading with different key names
        try:
            ckpt = torch.load(path, map_location=device)
            
            # Try different possible keys for model state
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=False)
            elif 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                # Assume the whole dict is the state dict
                model.load_state_dict(ckpt, strict=False)
            
            # Try to load optimizer if provided
            if optimizer is not None:
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                elif 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
            
            return ckpt
        except Exception as e2:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e2}") from e


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_acc: Optional[float] = None,
    best_val_acc: Optional[float] = None,
    **extra: Any
) -> None:
    """
    Save a checkpoint with model and optional optimizer state.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        val_acc: Optional validation accuracy
        best_val_acc: Optional best validation accuracy
        **extra: Additional metadata to save
    """
    extra_data = {
        'epoch': epoch,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        **extra
    }
    
    # Remove None values
    extra_data = {k: v for k, v in extra_data.items() if v is not None}
    
    _save_checkpoint_base(
        path,
        model,
        optimizer,
        extra_data if extra_data else None
    )


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """
    Extract metadata from a checkpoint file without loading the full model.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata (epoch, val_acc, etc.)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location='cpu')
    
    info = {}
    
    # Extract common metadata fields
    for key in ['epoch', 'val_acc', 'best_val_acc', 'train_loss', 'val_loss', 
                'train_acc', 'best_epoch', 'model_state_dict']:
        if key in ckpt:
            if key == 'model_state_dict':
                # Just note that it exists, don't load it
                info['has_model'] = True
            else:
                info[key] = ckpt[key]
    
    return info

