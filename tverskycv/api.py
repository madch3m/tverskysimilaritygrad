#!/usr/bin/env python3
"""
Simple user-friendly API for TverskyCV.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .training.entry_points import train_from_config
from .training.model_builder import build_model_from_config
from .training.config_utils import load_config, validate_config
from .registry import DATASETS


def train(config: str, **kwargs) -> Dict[str, Any]:
    """
    Train a model from a configuration file.
    
    Args:
        config: Path to YAML configuration file
        **kwargs: Additional arguments (device, checkpoint, use_optimized, etc.)
        
    Returns:
        Dictionary with training results
        
    Example:
        >>> results = train("config.yaml")
        >>> print(f"Best accuracy: {results['best_val_acc']:.4f}")
    """
    return train_from_config(config, **kwargs)


def build_model(config: str, device: Optional[torch.device] = None) -> nn.Module:
    """
    Build a model from a configuration file.
    
    Args:
        config: Path to YAML configuration file
        device: Optional device to place model on (default: auto-detect)
        
    Returns:
        Initialized PyTorch model
        
    Example:
        >>> model = build_model("config.yaml")
        >>> model.eval()
    """
    cfg = load_config(config)
    validate_config(cfg)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return build_model_from_config(cfg, device)


def evaluate(
    model: nn.Module,
    config: str,
    split: str = 'val'
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Trained PyTorch model
        config: Path to YAML configuration file (for dataset info)
        split: Dataset split to evaluate on ('train' or 'val')
        
    Returns:
        Dictionary with evaluation metrics (accuracy, loss, etc.)
        
    Example:
        >>> model = build_model("config.yaml")
        >>> results = evaluate(model, "config.yaml", split='val')
        >>> print(f"Validation accuracy: {results['accuracy']:.4f}")
    """
    from .training.engine import evaluate as _evaluate
    from .training.setup import setup_training_from_config
    
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset_name = cfg['dataset']['name']
    dataset_params = cfg['dataset'].get('params', {})
    dm = DATASETS.get(dataset_name)(**dataset_params)
    
    # Setup criterion
    setup = setup_training_from_config(cfg, model)
    criterion = setup['criterion']
    
    # Get appropriate dataloader
    if split == 'val':
        dataloader = dm.val_dataloader()
    elif split == 'train':
        dataloader = dm.train_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'")
    
    # Evaluate
    accuracy = _evaluate(model, dataloader, device, criterion)
    
    return {
        'accuracy': accuracy,
        'split': split
    }

