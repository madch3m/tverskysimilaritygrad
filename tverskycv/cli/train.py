# tverskycv/cli/train.py
import argparse
import yaml
import torch
import torch.nn as nn

from tverskycv.registry import BACKBONES, HEADS, DATASETS
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.training.engine import fit
from tverskycv.training.utils import set_seed, resolve_device
from tverskycv.training.optimizers import build_optimizer
from tverskycv.training.schedulers import build_scheduler


def main():
    parser = argparse.ArgumentParser(description="Train a TverskyCV model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # -----------------------
    # Load config
    # -----------------------
    cfg = yaml.safe_load(open(args.config, "r"))
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # -----------------------
    # Dataset
    # -----------------------
    dataset_name = cfg["dataset"]["name"]
    dataset_params = cfg["dataset"].get("params", {})
    dm = DATASETS.get(dataset_name)(**dataset_params)

    # -----------------------
    # Model
    # -----------------------
    backbone_cfg = cfg["model"]["backbone"]
    head_cfg = cfg["model"]["head"]

    backbone = BACKBONES.get(backbone_cfg["name"])(**backbone_cfg.get("params", {}))
    head = HEADS.get(head_cfg["name"])(**head_cfg.get("params", {}))
    model = ImageClassifier(backbone, head)

    device = resolve_device(cfg["train"].get("device", None))
    model.to(device)

    # -----------------------
    # Optimizer + Scheduler
    # -----------------------
    optimizer_cfg = cfg.get("optimizer", {"name": "adamw", "params": {"lr": cfg["train"]["lr"]}})
    optimizer = build_optimizer(model.parameters(), optimizer_cfg)

    scheduler_cfg = cfg["train"].get("scheduler", None)
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    # -----------------------
    # Criterion
    # -----------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training
    # -----------------------
    print(f"🚀 Starting training on {device} ...")

    stats = fit(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg["train"]["epochs"],
        scheduler=scheduler,
        ckpt_dir=cfg["train"]["ckpt_dir"],
    )

    print(f"Best validation accuracy: {stats['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
