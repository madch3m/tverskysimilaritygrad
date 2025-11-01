import argparse
import yaml
import torch

from ..registry.registry import BACKBONES, HEADS, DATASETS
from ..models.wrappers.classifiers import ImageClassifier
from training.engine import evaluate
from training.utils import set_seed


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TverskyCV model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config))

    # Set seed
    if "seed" in cfg:
        set_seed(cfg["seed"])

    device = cfg.get("train", {}).get("device", "cpu")

    # Build dataset
    dataset_name = cfg["dataset"]["name"]
    dataset_params = cfg["dataset"].get("params", {})
    datamodule = DATASETS.get(dataset_name)(**dataset_params)

    # Build backbone
    backbone_name = cfg["model"]["backbone"]["name"]
    backbone_params = cfg["model"]["backbone"].get("params", {})
    backbone = BACKBONES.get(backbone_name)(**backbone_params)

    # Build head
    head_name = cfg["model"]["head"]["name"]
    head_params = cfg["model"]["head"].get("params", {})
    head = HEADS.get(head_name)(**head_params)

    # Wrap model
    model = ImageClassifier(backbone, head).to(device)

    # Load checkpoint
    model = load_checkpoint(model, args.ckpt, device)

    # Evaluate
    val_loader = datamodule.val_dataloader()
    acc = evaluate(model, val_loader, device)

    print(f"✅ Final accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()