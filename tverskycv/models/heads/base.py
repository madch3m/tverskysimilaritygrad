from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn


def _ensure_2d(x: Tensor) -> Tensor:
    """
    Ensures features are shaped [B, D]. Flattens trailing dims if needed.
    Raises if input rank < 2 (i.e., no batch dim).
    """
    if x.ndim < 2:
        raise ValueError(
            f"Head expected input with rank >= 2 (B, D, ...), got shape={tuple(x.shape)}"
        )
    if x.ndim > 2:
        # Flatten all trailing feature dims into one vector per sample
        b = x.shape[0]
        x = x.view(b, -1)
    return x


class IProjectionHead(ABC, nn.Module):
    """
    Abstract base class for all projection heads.

    Implementors MUST:
      - accept a float Tensor of shape [B, D] (or [B, ...] that can be flattened to [B, D])
      - return a float Tensor of shape [B, C]
      - define output_dim() -> int that returns C

    Optional overrides:
      - reset_parameters(): re-initialize learnable params (called by trainers/tests if present)
      - extra_metadata(): return a small dict of head-specific info for logging
    """

    def __init__(self) -> None:
        super().__init__()
        # Subclasses should set this to the head's output dimension (C).
        self._out_dim: Optional[int] = None

    # --- required API ---

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Tensor with shape [B, D] or [B, ...] (will be flattened to [B, D] by subclasses
               using `_ensure_2d` prior to their own logic if desired).

        Returns:
            Tensor with shape [B, C] where C == output_dim().
        """
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns:
            The size of the last dimension of the head's output (C).
        """
        raise NotImplementedError

    # --- recommended helpers for consistency (optional to override) ---

    def ensure_2d(self, x: Tensor) -> Tensor:
        """
        Convenience wrapper so subclasses can call `self.ensure_2d(x)` uniformly.
        """
        return _ensure_2d(x)

    def reset_parameters(self) -> None:
        """
        Optional: (Re)initialize parameters. Subclasses can override.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / max(1, fan_in) ** 0.5
                    nn.init.uniform_(m.bias, -bound, bound)

    def extra_metadata(self) -> Dict[str, Any]:
        """
        Return lightweight, JSON-serializable metadata for logging.
        Subclasses can extend this dict.
        """
        return {
            "head_class": self.__class__.__name__,
            "output_dim": self.output_dim_safe(),
            "num_params": sum(p.numel() for p in self.parameters()),
        }

    # --- convenience accessors ---

    def output_dim_safe(self) -> Optional[int]:
        """
        Returns output_dim if available without forcing subclasses to compute anything heavy.
        Useful for logging when subclasses set `self._out_dim` during __init__.
        """
        try:
            # Prefer cached _out_dim if the subclass set it
            if self._out_dim is not None:
                return int(self._out_dim)
            # Fall back to the abstract method (may raise if not initialized yet)
            return int(self.output_dim())
        except Exception:
            return None

    @property
    def num_classes(self) -> Optional[int]:
        """
        Alias for output_dim in classification use-cases; may be None if not yet initialized.
        """
        return self.output_dim_safe()

    def extra_repr(self) -> str:
        od = self.output_dim_safe()
        return f"output_dim={od if od is not None else 'unset'}"
