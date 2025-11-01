import argparse, yaml, torch, torch.nn as nn, torch.optim as optim
from ..registry.registry import BACKBONES, HEADS, DATASETS
from ..models.wrappers.classifiers import ImageClassifier
from training.engine import train_one_epoch, evaluate
from training.utils import set_seed, save_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tverskycv/configs/mnist.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    dm = DATASETS.get(cfg["dataset"]["name"])(**cfg["dataset"]["params"])
    backbone = BACKBONES.get(cfg["model"]["backbone"]["name"])(**cfg["model"]["backbone"]["params"])
    head = HEADS.get(cfg["model"]["head"]["name"])(**cfg["model"]["head"]["params"])
    model = ImageClassifier(backbone, head).to(cfg["train"]["device"])

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        train_one_epoch(model, dm.train_dataloader(), opt, criterion, cfg["train"]["device"])
        acc = evaluate(model, dm.val_dataloader(), cfg["train"]["device"])
        if acc > best:
            best = acc
            save_ckpt(model, opt, acc, cfg["train"]["ckpt_dir"])
    print(f"best_acc={best:.4f}")

if __name__ == "__main__":
    main()