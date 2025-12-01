import os
import random
import torch
import numpy as np
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Might slow training a little, but best reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def save_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


# ============================================================================
# Transfer Learning Utilities
# ============================================================================

def freeze_model(model: torch.nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_backbone(model: torch.nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze backbone layers in a model.
    
    Assumes model has a 'feature_extractor' or 'backbone' attribute.
    """
    if hasattr(model, 'feature_extractor'):
        freeze_model(model.feature_extractor, freeze)
    elif hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'feature_extractor'):
            freeze_model(model.backbone.feature_extractor, freeze)
        else:
            freeze_model(model.backbone, freeze)
    else:
        # Try to find common backbone names
        for name, module in model.named_children():
            if 'backbone' in name.lower() or 'feature' in name.lower() or 'encoder' in name.lower():
                freeze_model(module, freeze)
                break


def get_trainable_params(model: torch.nn.Module) -> int:
    """Get count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model: torch.nn.Module) -> int:
    """Get count of all parameters."""
    return sum(p.numel() for p in model.parameters())


def get_layer_names(model: torch.nn.Module, prefix: str = '') -> list:
    """Get list of all layer names in a model."""
    layers = []
    for name, _ in model.named_modules():
        if prefix:
            layers.append(f"{prefix}.{name}" if name else prefix)
        else:
            layers.append(name)
    return layers


def unfreeze_layers_progressively(
    model: torch.nn.Module,
    unfreeze_ratio: float,
    layer_prefix: str = 'feature_extractor'
) -> int:
    """
    Progressively unfreeze layers in a model based on ratio.
    
    Args:
        model: The model to unfreeze layers in
        unfreeze_ratio: Ratio of layers to unfreeze (0.0 to 1.0)
        layer_prefix: Prefix to identify layers to unfreeze (e.g., 'feature_extractor')
    
    Returns:
        Number of layers unfrozen
    """
    # Find the module to unfreeze
    target_module = None
    if hasattr(model, layer_prefix):
        target_module = getattr(model, layer_prefix)
    elif hasattr(model, 'backbone') and hasattr(model.backbone, layer_prefix):
        target_module = getattr(model.backbone, layer_prefix)
    
    if target_module is None:
        return 0
    
    # Get all named modules in the target
    named_modules = list(target_module.named_modules())
    if len(named_modules) == 0:
        return 0
    
    # Calculate how many layers to unfreeze
    num_to_unfreeze = int(len(named_modules) * unfreeze_ratio)
    
    # Unfreeze from the end (top layers first)
    unfrozen_count = 0
    for i, (name, module) in enumerate(reversed(named_modules)):
        if i < num_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
            unfrozen_count += 1
    
    return unfrozen_count


class ProgressiveUnfreezing:
    """
    Manages progressive unfreezing of model layers during training.
    
    Example:
        schedule = {
            0: 0.0,      # Epoch 0-4: Freeze all backbone
            5: 0.33,     # Epoch 5-9: Unfreeze top 33%
            10: 0.66,    # Epoch 10-14: Unfreeze top 66%
            15: 1.0      # Epoch 15+: Unfreeze all
        }
        unfreezer = ProgressiveUnfreezing(model, schedule)
        
        for epoch in range(num_epochs):
            unfreezer.unfreeze_for_epoch(epoch)
            train_one_epoch(...)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        schedule: dict,
        layer_prefix: str = 'feature_extractor'
    ):
        """
        Initialize progressive unfreezing manager.
        
        Args:
            model: Model to manage unfreezing for
            schedule: Dictionary mapping epoch -> unfreeze_ratio
            layer_prefix: Prefix to identify layers to unfreeze
        """
        self.model = model
        self.schedule = schedule
        self.layer_prefix = layer_prefix
        self.current_ratio = 0.0
        self.last_epoch = -1
        
        # Initially freeze backbone
        freeze_backbone(model, freeze=True)
    
    def unfreeze_for_epoch(self, epoch: int) -> float:
        """
        Unfreeze layers based on current epoch.
        
        Args:
            epoch: Current training epoch
        
        Returns:
            Current unfreeze ratio
        """
        # Find the appropriate ratio for this epoch
        ratio = 0.0
        for schedule_epoch in sorted(self.schedule.keys(), reverse=True):
            if epoch >= schedule_epoch:
                ratio = self.schedule[schedule_epoch]
                break
        
        # Only update if ratio changed
        if ratio != self.current_ratio or epoch != self.last_epoch:
            if ratio > 0.0:
                unfrozen = unfreeze_layers_progressively(
                    self.model,
                    ratio,
                    self.layer_prefix
                )
                self.current_ratio = ratio
                self.last_epoch = epoch
                return ratio
        
        return self.current_ratio
    
    def get_trainable_info(self) -> dict:
        """Get information about trainable parameters."""
        total = get_total_params(self.model)
        trainable = get_trainable_params(self.model)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_ratio': trainable / total if total > 0 else 0.0,
            'unfreeze_ratio': self.current_ratio
        }