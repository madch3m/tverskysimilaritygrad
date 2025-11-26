from __future__ import annotations

from typing import Dict, Any, Optional
from math import cos, pi
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def build_scheduler(optimizer: Optimizer, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}
    name = cfg.get("name", "none").lower()
    p = cfg.get("params", {}) or {}

    if name in ("none", "", None):
        return None

    if name == "step":
        step_size = p.get("step_size", 5)
        gamma = p.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        tmax = p.get("T_max", 10)
        eta_min = p.get("eta_min", 0.0)
        return CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)

    if name == "cosine_warmup":
        return _CosineWarmupScheduler(optimizer, **p)

    raise ValueError(f"Unknown scheduler '{name}'")


class _CosineWarmupScheduler:
    _by_step = True

    def __init__(self, optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if self.step_num <= self.warmup_steps:
                # linear warmup
                lr = base * self.step_num / self.warmup_steps
            else:
                # cosine decay
                t = (self.step_num - self.warmup_steps) / max(1, (self.max_steps - self.warmup_steps))
                lr = self.min_lr + (base - self.min_lr) * 0.5 * (1 + cos(pi * t))
            pg["lr"] = lr
