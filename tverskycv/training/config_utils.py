#!/usr/bin/env python3
"""
Configuration utilities for unified config loading and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import Namespace


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return cfg


def resolve_training_config(cfg: Dict[str, Any], args: Optional[Namespace] = None) -> Dict[str, Any]:
    """
    Merge CLI arguments with config file, with CLI args taking precedence.
    
    Args:
        cfg: Configuration dictionary from YAML
        args: Optional argparse Namespace with CLI arguments
        
    Returns:
        Merged configuration dictionary
    """
    if args is None:
        return cfg
    
    # Override device if provided in args
    if hasattr(args, 'device') and args.device is not None:
        if 'train' not in cfg:
            cfg['train'] = {}
        cfg['train']['device'] = args.device
    
    # Override checkpoint path if provided
    if hasattr(args, 'ckpt') and args.ckpt is not None:
        cfg['checkpoint_path'] = args.ckpt
    
    # Override log directory if provided
    if hasattr(args, 'log_dir') and args.log_dir is not None:
        if 'logging' not in cfg:
            cfg['logging'] = {}
        cfg['logging']['log_dir'] = args.log_dir
    
    # Override epochs if provided
    if hasattr(args, 'epochs') and args.epochs is not None:
        if 'train' not in cfg:
            cfg['train'] = {}
        cfg['train']['epochs'] = args.epochs
    
    return cfg


def validate_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate that required configuration fields are present.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        ('dataset', 'name'),
        ('model', 'backbone', 'name'),
        ('model', 'head', 'name'),
        ('train', 'epochs'),
        ('train', 'lr'),
    ]
    
    missing_fields = []
    for field_path in required_fields:
        current = cfg
        path_str = '.'.join(field_path)
        try:
            for field in field_path:
                current = current[field]
            if current is None:
                missing_fields.append(path_str)
        except (KeyError, TypeError):
            missing_fields.append(path_str)
    
    if missing_fields:
        raise ValueError(
            f"Missing required configuration fields: {', '.join(missing_fields)}\n"
            f"Please ensure your config file includes all required fields."
        )
    
    return True

