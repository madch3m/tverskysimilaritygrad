#!/usr/bin/env python3
"""
Unified model building utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from tverskycv.registry import BACKBONES, HEADS
from tverskycv.models.wrappers.classifiers import ImageClassifier


def build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build a model from configuration dictionary.
    
    Args:
        cfg: Configuration dictionary with 'model' key containing 'backbone' and 'head'
        device: Target device for the model
        
    Returns:
        Initialized model on the specified device
        
    Raises:
        KeyError: If required config fields are missing
        ValueError: If backbone or head name is not registered
    """
    if 'model' not in cfg:
        raise KeyError("Config must contain 'model' key")
    
    model_cfg = cfg['model']
    
    # Build backbone
    if 'backbone' not in model_cfg:
        raise KeyError("Config must contain 'model.backbone' key")
    
    backbone_cfg = model_cfg['backbone']
    backbone_name = backbone_cfg['name']
    backbone_params = backbone_cfg.get('params', {})
    
    if backbone_name not in BACKBONES._fns:
        raise ValueError(f"Backbone '{backbone_name}' not found in registry. Available: {list(BACKBONES._fns.keys())}")
    
    backbone = BACKBONES.get(backbone_name)(**backbone_params)
    
    # Build head
    if 'head' not in model_cfg:
        raise KeyError("Config must contain 'model.head' key")
    
    head_cfg = model_cfg['head']
    head_name = head_cfg['name']
    head_params = head_cfg.get('params', {})
    
    if head_name not in HEADS._fns:
        raise ValueError(f"Head '{head_name}' not found in registry. Available: {list(HEADS._fns.keys())}")
    
    head = HEADS.get(head_name)(**head_params)
    
    # Create and move model to device
    model = ImageClassifier(backbone, head).to(device)
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

