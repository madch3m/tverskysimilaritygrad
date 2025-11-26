import argparse
import os
from pathlib import Path
import random

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tverskycv.registry import BACKBONES, HEADS, DATASETS
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.training.engine import evaluate
from tverskycv.training.utils import set_seed, resolve_device

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def collect_logits_labels_images(model, dataloader, device, max_batches=None):
    model.eval()
    all_logits, all_labels, all_images = [], [], []
    for bi, (xb, yb) in enumerate(dataloader):
        xb = xb.to(device)
        logits = model(xb).cpu()
        all_logits.append(logits)
        all_labels.append(yb.cpu())
        all_images.append(xb.cpu())
        if max_batches is not None and (bi + 1) >= max_batches:
            break

    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_images)


def confusion_matrix_torch(preds, labels, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm):
    correct = cm.diag().float()
    totals = cm.sum(dim=1).float().clamp_min(1)
    return (correct / totals).numpy()


def print_per_class_acc(per_cls_acc):
    print("\nPer-class accuracy:")
    for i, a in enumerate(per_cls_acc):
        print(f"  class {i}: {a:.4f}")

    worst = sorted(range(len(per_cls_acc)), key=lambda i: per_cls_acc[i])[:3]
    print("\nWorst classes:", worst)


def plot_confusion_matrix(cm, class_names=None, normalize=False, title="Confusion Matrix", save_path=None):
    if not _HAS_PLT:
        print("matplotlib not installed; skipping confusion matrix plot.")
        return

    cm = cm.numpy().astype(np.float32)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, class_names if class_names else ticks)
    plt.yticks(ticks, class_names if class_names else ticks)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()


def plot_image_grid(images, labels, preds=None, title="", max_show=16, save_path=None):
    if not _HAS_PLT:
        print("matplotlib not installed; skipping image grid.")
        return

    n = min(max_show, len(images))
    cols = int(np.sqrt(max_show))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        img = images[i].squeeze(0)  # MNIST: [1,28,28] -> [28,28]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        if preds is None:
            plt.title(f"T:{int(labels[i])}")
        else:
            plt.title(f"T:{int(labels[i])} P:{int(preds[i])}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved grid to {save_path}")
    else:
        plt.show()


def plot_confidence_hist(confidences, save_path=None):
    if not _HAS_PLT:
        print("matplotlib not installed; skipping confidence histogram.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=30)
    plt.title("Prediction Confidence Histogram")
    plt.xlabel("Max softmax probability")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved confidence histogram to {save_path}")
    else:
        plt.show()


@torch.no_grad()
def plot_pca_embeddings(backbone, images, labels, device, max_points=2000, save_path=None):
    """
    Simple PCA visualization of backbone embeddings.
    No sklearn required.
    """
    if not _HAS_PLT:
        print("matplotlib not installed; skipping PCA embeddings.")
        return

    n = min(max_points, len(images))
    idx = torch.randperm(len(images))[:n]
    x = images[idx].to(device)
    y = labels[idx].numpy()

    feats = backbone(x).cpu().numpy()  # [n, D]

    # PCA via SVD
    feats = feats - feats.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(feats, full_matrices=False)
    z = feats @ Vt[:2].T  # [n, 2]

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(z[:, 0], z[:, 1], c=y, s=8)
    plt.title("Backbone Embeddings (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved PCA plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate + visualize a TverskyCV model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for faster eval")
    parser.add_argument("--show-correct", type=int, default=16, help="# correct examples to plot")
    parser.add_argument("--show-wrong", type=int, default=16, help="# wrong examples to plot")
    parser.add_argument("--normalize-cm", action="store_true", help="Normalize confusion matrix rows")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots")
    parser.add_argument("--pca", action="store_true", help="Plot PCA of backbone embeddings")
    parser.add_argument("--out-dir", default=None, help="If set, saves plots to this folder")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("train", {}).get("device", None))

    dm = DATASETS.get(cfg["dataset"]["name"])(**cfg["dataset"]["params"])
    val_loader = dm.val_dataloader()

    backbone = BACKBONES.get(cfg["model"]["backbone"]["name"])(**cfg["model"]["backbone"]["params"])
    head = HEADS.get(cfg["model"]["head"]["name"])(**cfg["model"]["head"]["params"])
    model = ImageClassifier(backbone, head).to(device)
    model = load_checkpoint(model, args.ckpt, device)

    num_classes = cfg["model"]["head"]["params"]["num_classes"]

    acc = evaluate(model, val_loader, device)
    print(f"\n✅ Final accuracy: {acc:.4f}")

    logits, labels, images = collect_logits_labels_images(
        model, val_loader, device, max_batches=args.max_batches
    )
    preds = logits.argmax(dim=-1)
    probs = F.softmax(logits, dim=-1)
    confs = probs.max(dim=-1).values.numpy()

    cm = confusion_matrix_torch(preds, labels, num_classes)
    print("\n🔢 Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    per_cls = per_class_accuracy(cm)
    print_per_class_acc(per_cls)

    wrong_idx = (preds != labels).nonzero().flatten()
    correct_idx = (preds == labels).nonzero().flatten()

    print(f"\nMisclassified: {len(wrong_idx)} / {len(labels)}")
    print(f"Correct:       {len(correct_idx)} / {len(labels)}")

    if args.no_plots:
        return

    out_dir = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        normalize=args.normalize_cm,
        title="Confusion Matrix" + (" (Normalized)" if args.normalize_cm else ""),
        save_path=str(out_dir / "confusion_matrix.png") if out_dir else None,
    )

    if len(correct_idx) > 0:
        samp = correct_idx[torch.randperm(len(correct_idx))[: args.show_correct]]
        plot_image_grid(
            images[samp],
            labels[samp],
            preds[samp],
            title="Random Correct Predictions",
            max_show=args.show_correct,
            save_path=str(out_dir / "correct_grid.png") if out_dir else None,
        )

    if len(wrong_idx) > 0:
        samp = wrong_idx[torch.randperm(len(wrong_idx))[: args.show_wrong]]
        plot_image_grid(
            images[samp],
            labels[samp],
            preds[samp],
            title="Misclassified Examples",
            max_show=args.show_wrong,
            save_path=str(out_dir / "wrong_grid.png") if out_dir else None,
        )

    plot_confidence_hist(
        confs,
        save_path=str(out_dir / "confidence_hist.png") if out_dir else None,
    )

    if args.pca:
        plot_pca_embeddings(
            backbone.to(device),
            images,
            labels,
            device,
            save_path=str(out_dir / "pca_embeddings.png") if out_dir else None,
        )


if __name__ == "__main__":
    main()
