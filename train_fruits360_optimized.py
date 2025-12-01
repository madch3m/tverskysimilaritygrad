#!/usr/bin/env python3
"""
Optimized Training Script for Fruits-360 Classification
Target: 90%+ accuracy on 113 classes

Key Optimizations:
1. Transfer Learning from ImageNet pretrained ResNet
2. Progressive unfreezing of backbone layers
3. Data augmentation (already in dataset)
4. Learning rate scheduling with warmup
5. Label smoothing
6. Mixed precision training
7. Gradient clipping
8. Early stopping
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Import your TverskyCV modules
from tverskycv.registry import BACKBONES, HEADS, DATASETS
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.training.utils import set_seed, resolve_device, save_checkpoint
from tverskycv.training.optimizers import build_optimizer
from tverskycv.training.schedulers import build_scheduler
from tverskycv.models.backbones.shared_tversky import GlobalFeature


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class ProgressiveUnfreezing:
    """
    Progressive unfreezing strategy for transfer learning.
    Gradually unfreeze layers from top to bottom.
    """
    
    def __init__(self, model, unfreeze_schedule=None):
        self.model = model
        # Default: unfreeze in 3 stages over first 15 epochs
        self.schedule = unfreeze_schedule or {0: 0, 5: 0.33, 10: 0.66, 15: 1.0}
        
    def unfreeze_layers(self, epoch):
        """Unfreeze percentage of layers based on epoch."""
        if epoch not in self.schedule:
            return
            
        unfreeze_ratio = self.schedule[epoch]
        
        if unfreeze_ratio == 0:
            # Freeze all backbone layers
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        elif unfreeze_ratio == 1.0:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Partial unfreezing
            layers = list(self.model.backbone.children())
            n_unfreeze = int(len(layers) * unfreeze_ratio)
            
            # Freeze bottom layers
            for layer in layers[:-n_unfreeze]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
            # Unfreeze top layers
            for layer in layers[-n_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        print(f"Epoch {epoch}: Unfroze {unfreeze_ratio*100:.0f}% of backbone layers")


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
            
        self.log_file_path = self.log_dir / log_file
        self.metrics_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
    def log(self, message: str, print_console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float, val_acc: float, lr: float):
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
    
    def save_metrics(self, filepath: str = 'metrics.json'):
        filepath = self.log_dir / filepath
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to {filepath}")
    
    def close(self):
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()
        self.log("Training logger closed")


def build_model_from_cfg(cfg: dict, device: torch.device) -> nn.Module:
    """Build model with proper initialization."""
    backbone_cfg = cfg["model"]["backbone"]
    head_cfg = cfg["model"]["head"]
    
    # Build backbone (with pretrained weights if available)
    backbone = BACKBONES.get(backbone_cfg["name"])(**backbone_cfg.get("params", {}))
    
    # Build classification head
    head = HEADS.get(head_cfg["name"])(**head_cfg.get("params", {}))
    
    model = ImageClassifier(backbone, head).to(device)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validation"
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        
        logits = model(xb)
        loss = criterion(logits, yb)
        
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(-1) == yb).sum().item()
        total += xb.size(0)
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{total_loss/total:.4f}',
            'acc': f'{total_correct/total:.4f}'
        })
    
    avg_loss = total_loss / total if total > 0 else 0.0
    avg_acc = total_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def train_optimized(
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
    use_amp: bool = True,
    gradient_clip: float = 1.0,
    early_stopping: Optional[EarlyStopping] = None,
    progressive_unfreezing: Optional[ProgressiveUnfreezing] = None,
) -> Dict[str, float]:
    """
    Optimized training loop with all enhancements.
    """
    best_val_acc = 0.0
    best_epoch = 0
    scaler = GradScaler() if use_amp else None
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Progressive unfreezing
        if progressive_unfreezing:
            progressive_unfreezing.unfreeze_layers(epoch)
        
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (xb, yb) in enumerate(pbar):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
            
            # Update per-batch scheduler
            if scheduler and hasattr(scheduler, "step") and getattr(scheduler, "_by_step", False):
                scheduler.step()
            
            # Track metrics
            bsz = xb.size(0)
            train_loss += loss.item() * bsz
            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_total += bsz
            
            # Update progress
            pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{train_correct/train_total:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, 
                                     desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        # Update per-epoch scheduler
        if scheduler and hasattr(scheduler, "step") and not getattr(scheduler, "_by_step", False):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        if logger:
            logger.log_epoch(epoch, avg_train_loss, avg_train_acc, 
                           val_loss, val_acc, current_lr)
        
        # Save checkpoints
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            if ckpt_dir:
                save_checkpoint(
                    f"{ckpt_dir}/best.pt",
                    model=model,
                    optimizer=optimizer,
                    extra={
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "train_acc": avg_train_acc
                    }
                )
                if logger:
                    logger.log(f"✓ New best model! Val Acc: {val_acc:.4f}")
        
        # Save periodic checkpoint
        if ckpt_dir and (epoch + 1) % 5 == 0:
            save_checkpoint(
                f"{ckpt_dir}/checkpoint_epoch_{epoch+1}.pt",
                model=model,
                optimizer=optimizer,
                extra={"epoch": epoch + 1, "val_acc": val_acc}
            )
        
        # Print summary
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:03d}/{epochs} ({epoch_time:.1f}s) | "
            f"Train: loss={avg_train_loss:.4f} acc={avg_train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )
        
        # Early stopping check
        if early_stopping and early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimized training for Fruits-360 classification"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", default=None, help="Resume from checkpoint")
    parser.add_argument("--log-dir", default="./logs", help="Log directory")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", default=None, help="Override device")
    args = parser.parse_args()
    
    # Load config
    cfg = yaml.safe_load(open(args.config))
    
    # Set seed
    set_seed(cfg.get("seed", 42))
    
    # Device
    device = resolve_device(args.device or cfg["train"].get("device"))
    
    # Logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(log_dir, use_tensorboard=True)
    
    logger.log("=" * 80)
    logger.log("OPTIMIZED FRUITS-360 TRAINING")
    logger.log("=" * 80)
    logger.log(f"Config: {args.config}")
    logger.log(f"Device: {device}")
    logger.log(f"Mixed Precision: {not args.no_amp}")
    logger.log("=" * 80)
    
    # Clear GlobalFeature bank
    GlobalFeature().clear()
    
    # Build model
    logger.log("Building model...")
    model = build_model_from_cfg(cfg, device)
    total_params, trainable_params = count_parameters(model)
    logger.log(f"✓ Total parameters: {total_params:,}")
    logger.log(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if resuming
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.log(f"✓ Loaded checkpoint: {args.ckpt}")
    
    # Dataset
    logger.log("Loading dataset...")
    dataset_cfg = cfg["dataset"]
    dm = DATASETS.get(dataset_cfg["name"])(**dataset_cfg.get("params", {}))
    logger.log(f"✓ Train samples: {len(dm.train):,}")
    logger.log(f"✓ Val samples: {len(dm.val):,}")
    
    # Optimizer
    optimizer_cfg = cfg.get("optimizer", {
        "name": "adamw",
        "params": {"lr": cfg["train"]["lr"]}
    })
    optimizer = build_optimizer(model.parameters(), optimizer_cfg)
    
    # Scheduler
    scheduler_cfg = cfg["train"].get("scheduler")
    scheduler = build_scheduler(optimizer, scheduler_cfg) if scheduler_cfg else None
    
    # Criterion with label smoothing
    label_smoothing = cfg.get("train", {}).get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Training enhancements
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
    progressive_unfreezing = ProgressiveUnfreezing(
        model,
        unfreeze_schedule={0: 0, 5: 0.33, 10: 0.66, 15: 1.0}
    )
    
    # Checkpoint directory
    ckpt_dir = cfg["train"].get("ckpt_dir", "./checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    # Training
    logger.log("=" * 80)
    logger.log(f"Starting training for {cfg['train']['epochs']} epochs")
    logger.log("=" * 80)
    
    start_time = time.time()
    
    stats = train_optimized(
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
        use_amp=not args.no_amp,
        gradient_clip=cfg.get("training", {}).get("gradient_clip", 1.0),
        early_stopping=early_stopping,
        progressive_unfreezing=progressive_unfreezing,
    )
    
    total_time = time.time() - start_time
    
    logger.log("=" * 80)
    logger.log("TRAINING COMPLETE!")
    logger.log("=" * 80)
    logger.log(f"Total time: {total_time/60:.2f} minutes")
    logger.log(f"Best Val Accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_epoch']})")
    logger.log(f"Best model: {ckpt_dir}/best.pt")
    logger.log("=" * 80)
    
    logger.save_metrics()
    logger.close()
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best accuracy: {stats['best_val_acc']*100:.2f}%")
    print(f"✓ Logs: {log_dir}")


if __name__ == "__main__":
    main()
