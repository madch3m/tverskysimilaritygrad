# 🧠 TVERSKY-SIMILARITY-GRAD  
_Modular, scalable computer-vision framework for experimenting with Tversky-based similarity models_

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Key Concepts](#key-concepts)
  - [Backbones](#backbones)
  - [Projection Heads](#projection-heads)
  - [Tversky Projection Layer](#tversky-projection-layer)
- [Architecture](#architecture)
  - [Directory Layout](#directory-layout)
  - [Design Principles](#design-principles)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluating](#evaluating)
  - [Swapping Models via Config](#swapping-models-via-config)
  - [Jupyter Notebook](#jupyter-notebook)
- [Configuration](#configuration)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## Overview
**TVERSKY-SIMILARITY-GRAD** is a modular and testable deep-learning codebase supporting research on **Tversky Projection Layers**—a newly proposed neural architecture for **psychologically plausible similarity** (Doumbouya et al., 2025).  

This repository enables:
✅ Plug-and-play evaluation of different projection heads  
✅ Simple model benchmarking on MNIST (extensible to other datasets)  
✅ Clean separation between **architecture**, **training**, and **research**  
✅ Reproducible, testable experimentation  

Your team can implement the Tversky layer internally while others focus on data, training, or analysis.

---

## Motivation
Standard deep learning layers (e.g., dot-product linear layers) model **geometric similarity**, which fails to capture key properties of human perception—especially asymmetry (“the son resembles the father” ≠ “the father resembles the son”).

The **Tversky contrast model** defines similarity in terms of:
- Common features
- Distinctive features
- Weighted asymmetry

This repository provides the scaffolding to:
- Build CV pipelines using Tversky similarity
- Compare against baselines (linear, cosine)
- Experiment easily with new backbones + datasets

---

## Key Concepts

### Backbones
Feature extractors such as:
- Simple CNN
- ResNet variations
- Vision Transformers (future)

Backbones output a feature vector of dimension `D`.

### Projection Heads
Heads take `[B, D] → [B, C]` and allow model comparison via simple configuration.

Built-in:
- `LinearHead`
- `TverskyProjectionHead` (scaffold)

All heads implement:
```python
IProjectionHead.forward(x)
IProjectionHead.output_dim()
```

### Tversky Projection Layer
Implements a differentiable version of:
```
S(a,b) = θ·f(A∩B) – α·f(A–B) – β·f(B–A)
```

---

## Architecture

### Directory Layout
```
TVERSKY-SIMILARITY-GRAD/
  configs/              # YAML configs
    mnist.yaml
    models/
      simple_cnn.yaml
      resnet18.yaml
    heads/
      linear.yaml
      tversky.yaml
  data/
    datamodules.py      # MNIST loaders
    transforms.py
  models/
    backbones/
      simple_cnn.py
      resnet.py
    heads/
      base.py
      linear_head.py
      tversky_head.py
    wrappers/
      classifier.py
  registry/
    registry.py
  training/
    engine.py
    metrics.py
    utils.py
  cli/
    train.py
    eval.py
  tests/
    test_heads.py
    test_backbones.py
    test_wrapper.py
    test_datamodule.py
  notebooks/
    TverskySimilarity.ipynb
  scripts/
    export_onnx.py
README.md
```

---

### Design Principles

✅ **Composable**  
Backbones and heads can be swapped without touching training code.

✅ **Testable**  
Unit tests isolate:
- data loading
- backbones
- heads
- wrapper behavior

✅ **Minimal Notebook Logic**  
Notebooks import the library; all logic stays in the codebase.

✅ **Config-Driven**  
Model selection uses `.yaml` config files.

✅ **Extendable**  
Researchers can use the registry to add new heads/datasets.

---

## Installation

```bash
git clone <your-repo-url>
cd TVERSKY-SIMILARITY_GRAD
pip install -r requirements.txt
```

---

## Usage

### Training

Basic:
```bash
python -m TVERSKY-SIMILARITY-GRAD.cli.train --config TVERSKY-SIMILARITY-GRAD/configs/mnist.yaml
```

Trained model checkpoints automatically save to:
```
./checkpoints/
```

### Evaluating
```bash
python -m TVERSKY-SIMILARITY-GRAD.cli.eval --config TVERSKY-SIMILARITY-GRAD/configs/mnist.yaml --ckpt path/to/model.pt
```

---

### Swapping Models via Config
Edit `TVERSKY-SIMILARITY-GRAD/configs/mnist.yaml`:

```yaml
model:
  backbone:
    name: simple_cnn
  head:
    name: tversky
```

No training code change needed.

---

### Jupyter Notebook
Example in:

```
notebooks/TverskySimilarity.ipynb
```

Usage:
```python
from TVERSKY-SIMILARITY-GRAD.cli.train import main
main()
```

---

## Configuration

Configs are YAML. Example:

```yaml
seed: 1337

dataset:
  name: mnist
  params:
    data_dir: ./data
    batch_size: 256

model:
  backbone:
    name: simple_cnn
    params:
      out_dim: 128
  head:
    name: linear
    params:
      in_dim: 128
      num_classes: 10

train:
  epochs: 10
  lr: 1e-3
  device: cuda
```

---

## Testing

Run all tests:
```bash
pytest -q
```

Tests include:
- `test_heads.py`
- `test_backbones.py`
- `test_wrapper.py`
- `test_datamodule.py`

Goal:  
✅ Validate shapes/gradients  
✅ Fail fast when configs are wrong  

---

## Development Workflow

1) Pick/modify config YAML  
2) Run training  
3) Swap heads/backbones as needed  
4) Inspect results / log metrics  
5) Iterate  

Branching model:
- `main` = stable
- `feat/*` = feature branches
- `exp/*` = experiments

---

## Roadmap

✅ Basic MNIST baseline  
✅ Linear projection head  
✅ Registry for head + backbone swapping  
✅ CLI + config system  
✅ Unit testing + CI  

🚧 Tversky layer implementation  
🚧 Visualization utilities for Tversky prototypes  
🚧 Add CIFAR-10 support  
🚧 Add Vision Transformers  
🚧 ONNX + Torch-Script export  
🚧 Model cards + benchmarks  

---

## References
- Doumbouya et al. (2025).  
  *Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity*  
  https://arxiv.org/abs/2506.11035  

- Tversky, A. (1977).  
  *Features of similarity*. Psychological Review.

---

## License
MIT License
