from __future__ import annotations

from typing import Optional, Dict, Any, Iterable
import torch
from torch import nn
from torch.utils.data import DataLoader

from tverskycv.training.metrics import accuracy
from tverskycv.training.utils import save_checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    criterion: Optional[nn.Module] = None,
) -> float:
    model.eval()
    device = torch.device(device)
    total_correct, total = 0, 0
    total_loss = 0.0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            total_loss += criterion(logits, yb).item() * xb.size(0)
        total_correct += (logits.argmax(-1) == yb).sum().item()
        total += xb.size(0)

    if criterion is not None and total > 0:
        avg_loss = total_loss / total  # not returned but handy to print if needed
    return (total_correct / total) if total else 0.0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    device = torch.device(device)

    running_loss, running_acc, n = 0.0, 0.0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler and hasattr(scheduler, "step") and getattr(scheduler, "_by_step", False):
            scheduler.step()  # per-batch schedulers

        bsz = xb.size(0)
        running_loss += loss.item() * bsz
        running_acc += accuracy(logits, yb) * bsz
        n += bsz

    return {
        "loss": (running_loss / n) if n else 0.0,
        "acc": (running_acc / n) if n else 0.0,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device = "cpu",
    epochs: int = 1,
    scheduler: Optional[Any] = None,
    ckpt_dir: Optional[str] = None,
) -> Dict[str, float]:
    best_val = 0.0
    device = torch.device(device)

    for epoch in range(epochs):
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler=scheduler
        )
        if scheduler and hasattr(scheduler, "step") and not getattr(scheduler, "_by_step", False):
            scheduler.step()  # per-epoch schedulers

        val_acc = evaluate(model, val_loader, device, criterion=None)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            if ckpt_dir:
                save_checkpoint(
                    f"{ckpt_dir}/best.pt",
                    model=model,
                    optimizer=optimizer,
                    extra={"epoch": epoch + 1, "val_acc": val_acc},
                )

    return {"best_val_acc": best_val}
