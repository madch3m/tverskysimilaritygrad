#!/usr/bin/env python3
"""
Training script for TverskyCV models with enhanced logging
Follows the same pattern as export_onnx.py using registry system
"""

import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None  # Will be handled in TrainingLogger

from tverskycv.registry import BACKBONES, HEADS, DATASETS
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.training.engine import train_one_epoch, evaluate
from tverskycv.training.utils import set_seed, resolve_device, save_checkpoint
from tverskycv.training.optimizers import build_optimizer
from tverskycv.training.schedulers import build_scheduler
from tverskycv.models.backbones.shared_tversky import GlobalFeature


class TrainingLogger:
    """Comprehensive logging for training progress."""
    
    def __init__(self, log_dir: Path, use_tensorboard: bool = True, log_file: str = 'training.log'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        else:
            self.writer = None
            if use_tensorboard and not TENSORBOARD_AVAILABLE:
                print("⚠ TensorBoard not available, logging to file only")
        
        self.log_file_path = self.log_dir / log_file
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def log(self, message: str, print_console: bool = True):
        """Log message to file and optionally console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float, val_acc: float, lr: float):
        """Log epoch metrics."""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['learning_rate'].append(lr)
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
    
    def log_batch(self, epoch: int, batch_idx: int, total_batches: int, 
                  loss: float, acc: float, lr: Optional[float] = None):
        """Log batch-level metrics."""
        if self.use_tensorboard and self.writer is not None:
            global_step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss, global_step)
            self.writer.add_scalar('Accuracy/Train_Batch', acc, global_step)
            if lr is not None:
                self.writer.add_scalar('Learning_Rate_Batch', lr, global_step)
    
    def log_model_weights(self, model: nn.Module, epoch: int):
        """Log model weight histograms."""
        if self.use_tensorboard and self.writer is not None:
            for name, param in model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param.cpu(), epoch)
    
    def save_metrics(self, filepath: str = 'metrics.json'):
        """Save metrics history to JSON."""
        filepath = self.log_dir / filepath
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to {filepath}")
    
    def close(self):
        """Close logger and flush all logs."""
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        self.log("Training logger closed")


def build_model_from_cfg(cfg: dict, device: torch.device) -> nn.Module:
    """Build backbone + head per config and return wrapped model on device."""
    backbone_cfg = cfg["model"]["backbone"]
    head_cfg = cfg["model"]["head"]
    
    backbone = BACKBONES.get(backbone_cfg["name"])(**backbone_cfg.get("params", {}))
    head = HEADS.get(head_cfg["name"])(**head_cfg.get("params", {}))
    
    model = ImageClassifier(backbone, head).to(device)
    model.eval()  # Will be set to train() in training loop
    return model


def maybe_load_checkpoint(model: nn.Module, ckpt_path: Optional[str], device: torch.device) -> Dict[str, Any]:
    """Load checkpoint if provided, return checkpoint dict or empty dict."""
    if not ckpt_path or not Path(ckpt_path).exists():
        return {}
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    return ckpt


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_with_logging(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    scheduler: Optional[Any] = None,
    ckpt_dir: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
    log_every: int = 50,
    resume_epoch: int = 0,
) -> Dict[str, float]:
    """
    Enhanced training loop with logging support.
    Based on tverskycv.training.engine.fit but with progress bars and logging.
    """
    best_val = 0.0
    best_epoch = 0
    device = torch.device(device)
    
    for epoch in range(resume_epoch, epochs):
        epoch_start_time = time.time()
        
        # Training phase with progress bar
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_idx, (xb, yb) in enumerate(pbar):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            if scheduler and hasattr(scheduler, "step") and getattr(scheduler, "_by_step", False):
                scheduler.step()  # per-batch schedulers
            
            bsz = xb.size(0)
            running_loss += loss.item() * bsz
            from tverskycv.training.metrics import accuracy
            running_acc += accuracy(logits, yb) * bsz
            n += bsz
            
            # Update progress bar
            current_loss = running_loss / n
            current_acc = running_acc / n
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
            
            # Log batch metrics
            if logger and (batch_idx + 1) % log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.log_batch(epoch, batch_idx, len(train_loader), current_loss, current_acc, lr)
        
        train_loss = (running_loss / n) if n else 0.0
        train_acc = (running_acc / n) if n else 0.0
        
        # Update scheduler for per-epoch schedulers
        if scheduler and hasattr(scheduler, "step") and not getattr(scheduler, "_by_step", False):
            scheduler.step()
        
        # Validation phase with progress bar
        val_loss, val_acc = evaluate_with_progress(model, val_loader, criterion, device, epoch, epochs)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        if logger:
            logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Log model weights periodically
            if (epoch + 1) % 5 == 0:
                logger.log_model_weights(model, epoch)
        
        # Save checkpoint
        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc
            best_epoch = epoch + 1
        
        if ckpt_dir:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val,
            }
            
            # Save best model
            if is_best:
                save_checkpoint(
                    f"{ckpt_dir}/best.pt",
                    model=model,
                    optimizer=optimizer,
                    extra={"epoch": epoch + 1, "val_acc": val_acc},
                )
                if logger:
                    logger.log(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    f"{ckpt_dir}/checkpoint_epoch_{epoch+1}.pt",
                    model=model,
                    optimizer=optimizer,
                    extra={"epoch": epoch + 1, "val_acc": val_acc},
                )
            
            # Save latest checkpoint
            save_checkpoint(
                f"{ckpt_dir}/latest.pt",
                model=model,
                optimizer=optimizer,
                extra={"epoch": epoch + 1, "val_acc": val_acc},
            )
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1:03d}/{epochs} ({epoch_time:.2f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )
        if is_best and logger:
            logger.log(f"  ✓ Best model so far!")
    
    return {"best_val_acc": best_val, "best_epoch": best_epoch}


@torch.no_grad()
def evaluate_with_progress(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Evaluate with progress bar."""
    model.eval()
    device = torch.device(device)
    total_correct, total = 0, 0
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if criterion is not None:
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(-1) == yb).sum().item()
        total += xb.size(0)
        
        # Update progress bar
        current_loss = total_loss / total if total > 0 else 0.0
        current_acc = total_correct / total if total > 0 else 0.0
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
    
    avg_loss = (total_loss / total) if total > 0 else 0.0
    avg_acc = (total_correct / total) if total > 0 else 0.0
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train TverskyCV model with enhanced logging")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint .pt to resume from")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--log-every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--device", default=None, help="Override device from config")
    args = parser.parse_args()
    
    # Load config
    cfg = yaml.safe_load(open(args.config, "r"))
    
    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    # Resolve device
    device = resolve_device(args.device or cfg["train"].get("device", None))
    
    # Setup logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    use_tensorboard = cfg.get("logging", {}).get("use_tensorboard", True) and not args.no_tensorboard
    logger = TrainingLogger(log_dir, use_tensorboard=use_tensorboard)
    
    logger.log("=" * 70)
    logger.log("Training Configuration")
    logger.log("=" * 70)
    logger.log(f"Config file: {args.config}")
    logger.log(f"Device: {device}")
    logger.log(f"Seed: {seed}")
    logger.log(f"TensorBoard: {use_tensorboard}")
    logger.log("=" * 70)
    
    # Clear GlobalFeature bank if using shared features
    gf = GlobalFeature()
    gf.clear()
    
    # Build model from config (same pattern as export_onnx.py)
    logger.log("Building model from config...")
    model = build_model_from_cfg(cfg, device)
    num_params = count_parameters(model)
    logger.log(f"✓ Model created with {num_params:,} trainable parameters")
    
    # Load checkpoint if provided
    resume_epoch = 0
    if args.ckpt:
        logger.log(f"Loading checkpoint: {args.ckpt}")
        ckpt = maybe_load_checkpoint(model, args.ckpt, device)
        if ckpt:
            resume_epoch = ckpt.get("epoch", 0)
            logger.log(f"✓ Resumed from epoch {resume_epoch}")
    
    # Dataset (using registry)
    logger.log("Loading dataset...")
    dataset_name = cfg["dataset"]["name"]
    dataset_params = cfg["dataset"].get("params", {})
    dm = DATASETS.get(dataset_name)(**dataset_params)
    logger.log(f"✓ Dataset loaded: {len(dm.train)} train, {len(dm.val)} val samples")
    
    # Optimizer + Scheduler
    optimizer_cfg = cfg.get("optimizer", {"name": "adamw", "params": {"lr": cfg["train"]["lr"]}})
    optimizer = build_optimizer(model.parameters(), optimizer_cfg)
    
    scheduler_cfg = cfg["train"].get("scheduler", None)
    scheduler = build_scheduler(optimizer, scheduler_cfg)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Checkpoint directory
    ckpt_dir = cfg["train"].get("ckpt_dir", "./checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    # Training
    logger.log("=" * 70)
    logger.log(f"Starting training for {cfg['train']['epochs']} epochs")
    logger.log("=" * 70)
    
    start_time = time.time()
    
    stats = train_with_logging(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg["train"]["epochs"],
        scheduler=scheduler,
        ckpt_dir=ckpt_dir,
        logger=logger,
        log_every=args.log_every,
        resume_epoch=resume_epoch,
    )
    
    total_time = time.time() - start_time
    
    logger.log("=" * 70)
    logger.log("Training Complete!")
    logger.log("=" * 70)
    logger.log(f"Total time: {total_time/60:.2f} minutes")
    logger.log(f"Best Validation Accuracy: {stats['best_val_acc']:.4f} at epoch {stats['best_epoch']}")
    logger.log(f"Best model saved to: {ckpt_dir}/best.pt")
    logger.log(f"TensorBoard logs: {log_dir / 'tensorboard'}")
    
    # Save metrics
    logger.save_metrics()
    logger.close()
    
    print(f"\n✓ Training complete! View logs at: {log_dir}")
    if use_tensorboard:
        print(f"✓ TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
