#!/usr/bin/env python3
"""
Training script for TverskyCV models with enhanced logging
Uses unified training utilities for consistency
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use unified utilities
from tverskycv.training.config_utils import load_config, validate_config
from tverskycv.training.model_builder import build_model_from_config, count_parameters
from tverskycv.training.setup import setup_training_from_config
from tverskycv.training.checkpoint import load_checkpoint
from tverskycv.training.logging import TrainingLogger
from tverskycv.training.utils import set_seed
from tverskycv.registry import DATASETS
from tverskycv.training.engine import train_one_epoch, evaluate
from tverskycv.training.checkpoint import save_checkpoint

# TrainingLogger, build_model_from_cfg, and count_parameters are now in unified modules


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
    
    # Load and validate config using unified utilities
    cfg = load_config(args.config)
    validate_config(cfg)
    
    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    # Setup logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    use_tensorboard = cfg.get("logging", {}).get("use_tensorboard", True) and not args.no_tensorboard
    logger = TrainingLogger(log_dir, use_tensorboard=use_tensorboard)
    
    logger.log("=" * 70)
    logger.log("Training Configuration")
    logger.log("=" * 70)
    logger.log(f"Config file: {args.config}")
    logger.log(f"Device: {args.device or cfg.get('train', {}).get('device', 'auto')}")
    logger.log(f"Seed: {seed}")
    logger.log(f"TensorBoard: {use_tensorboard}")
    logger.log("=" * 70)
    
    # Build model using unified utilities
    logger.log("Building model from config...")
    device = torch.device(args.device or cfg.get('train', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model_from_config(cfg, device)
    total_params, trainable_params = count_parameters(model)
    logger.log(f"✓ Model created: {total_params:,} total, {trainable_params:,} trainable parameters")
    
    # Setup training components using unified utilities
    setup = setup_training_from_config(cfg, model, args)
    optimizer = setup['optimizer']
    scheduler = setup['scheduler']
    criterion = setup['criterion']
    ckpt_dir = setup['checkpoint_dir']
    epochs = setup['epochs']
    
    # Load checkpoint if provided
    resume_epoch = 0
    if args.ckpt:
        logger.log(f"Loading checkpoint: {args.ckpt}")
        ckpt = load_checkpoint(args.ckpt, model, device, optimizer)
        if ckpt:
            resume_epoch = ckpt.get("epoch", 0)
            logger.log(f"✓ Resumed from epoch {resume_epoch}")
    
    # Dataset (using registry)
    logger.log("Loading dataset...")
    dataset_name = cfg["dataset"]["name"]
    dataset_params = cfg["dataset"].get("params", {})
    dm = DATASETS.get(dataset_name)(**dataset_params)
    logger.log(f"✓ Dataset loaded: {len(dm.train)} train, {len(dm.val)} val samples")
    
    # Training
    logger.log("=" * 70)
    logger.log(f"Starting training for {epochs} epochs")
    logger.log("=" * 70)
    
    start_time = time.time()
    
    stats = train_with_logging(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
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
