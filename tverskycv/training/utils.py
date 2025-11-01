import os
import random
import torch
import numpy as np
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Might slow training a little, but best reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    """
    Convert string/None → torch.device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def save_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Save checkpoint.
    format:
        {
            "model": state_dict,
            "optimizer": state_dict,
            **extra
        }
    """
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
    """
    Load checkpoint state into model (+ optimizer optionally).
    Returns ckpt dict for logging etc.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy on (logits, targets).
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0