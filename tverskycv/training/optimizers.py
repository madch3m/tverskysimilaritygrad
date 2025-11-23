from __future__ import annotations

from typing import Dict, Any
import torch


def build_optimizer(params, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = (cfg or {}).get("name", "adamw").lower()
    p = (cfg or {}).get("params", {}) or {}

    if name in ("sgd",):
        lr = p.get("lr", 1e-2)
        momentum = p.get("momentum", 0.9)
        wd = p.get("weight_decay", 0.0)
        nesterov = p.get("nesterov", False)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)

    if name in ("adam",):
        lr = p.get("lr", 1e-3)
        betas = tuple(p.get("betas", (0.9, 0.999)))
        eps = p.get("eps", 1e-8)
        wd = p.get("weight_decay", 0.0)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    if name in ("adamw",):
        lr = p.get("lr", 1e-3)
        betas = tuple(p.get("betas", (0.9, 0.999)))
        eps = p.get("eps", 1e-8)
        wd = p.get("weight_decay", 1e-4)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    raise ValueError(f"Unknown optimizer '{name}'")
