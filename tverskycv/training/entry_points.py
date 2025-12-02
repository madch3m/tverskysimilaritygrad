#!/usr/bin/env python3
"""
Unified training entry points.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace

import torch
import torch.nn as nn

from .config_utils import load_config, resolve_training_config, validate_config
from .model_builder import build_model_from_config, count_parameters
from .setup import setup_training_from_config
from .checkpoint import load_checkpoint
from .utils import set_seed, resolve_device
from .engine import fit
from tverskycv.registry import DATASETS
from tverskycv.models.backbones.shared_tversky import GlobalFeature


def train_from_config(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    resume_epoch: Optional[int] = None,
    use_optimized_trainer: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified training function that handles all setup and training.
    
    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Optional path to checkpoint to resume from
        resume_epoch: Optional epoch to resume from (if different from checkpoint)
        use_optimized_trainer: Whether to use OptimizedTrainer instead of basic fit()
        **kwargs: Additional arguments to override config (device, log_dir, etc.)
        
    Returns:
        Dictionary with training statistics:
            - best_val_acc: Best validation accuracy
            - best_epoch: Epoch with best accuracy
            - total_epochs: Total epochs trained
            - train_losses: List of training losses per epoch
            - val_losses: List of validation losses per epoch
            - train_accs: List of training accuracies per epoch
            - val_accs: List of validation accuracies per epoch
    """
    # Load and validate config
    cfg = load_config(config_path)
    validate_config(cfg)
    
    # Set seed
    seed = cfg.get('seed', 42)
    set_seed(seed)
    
    # Clear GlobalFeature bank if using shared features
    GlobalFeature().clear()
    
    # Build model
    device = resolve_device(kwargs.get('device') or cfg.get('train', {}).get('device'))
    model = build_model_from_config(cfg, device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"✓ Model created: {total_params:,} total, {trainable_params:,} trainable parameters")
    
    # Setup training components
    # Create a mock args object from kwargs
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    mock_args = MockArgs(**kwargs)
    setup = setup_training_from_config(cfg, model, mock_args)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path, model, setup['device'], setup['optimizer'])
        if ckpt:
            start_epoch = ckpt.get('epoch', resume_epoch or 0)
            print(f"✓ Resumed from epoch {start_epoch}")
    
    # Load dataset
    dataset_name = cfg['dataset']['name']
    dataset_params = cfg['dataset'].get('params', {})
    dm = DATASETS.get(dataset_name)(**dataset_params)
    print(f"✓ Dataset loaded: {len(dm.train)} train, {len(dm.val)} val samples")
    
    # Choose training method
    if use_optimized_trainer:
        # Use OptimizedTrainer for advanced features
        from .optimized_trainer import OptimizedTrainer, create_optimized_dataloaders
        
        train_loader, val_loader = create_optimized_dataloaders(
            dm.train,
            dm.val,
            batch_size=dataset_params.get('batch_size')
        )
        
        trainer = OptimizedTrainer(
            model=model,
            device=setup['device'],
            num_epochs=setup['epochs'],
            learning_rate=setup['learning_rate'],
            checkpoint_dir=setup['checkpoint_dir'],
            **kwargs
        )
        
        results = trainer.train(train_loader, val_loader)
        return results
    else:
        # Use basic fit() function
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        stats = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=setup['optimizer'],
            criterion=setup['criterion'],
            device=setup['device'],
            epochs=setup['epochs'],
            scheduler=setup['scheduler'],
            ckpt_dir=setup['checkpoint_dir']
        )
        
        return {
            'best_val_acc': stats.get('best_val_acc', 0.0),
            'best_epoch': stats.get('best_epoch', setup['epochs']),
            'total_epochs': setup['epochs']
        }

