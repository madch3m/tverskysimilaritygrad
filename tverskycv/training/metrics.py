from __future__ import annotations

from typing import Iterable, Tuple
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1,)
) -> Tuple[float, ...]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [maxk, B]
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(float(correct_k.mul_(1.0 / logits.size(0)).item()))
    return tuple(res)
