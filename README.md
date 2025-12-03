# 🧠 TVERSKY-SIMILARITY-GRAD

_Modular, scalable computer-vision framework for experimenting with Tversky-based similarity models_

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Methods](#usage-methods)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Python API](#python-api)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Multi-GPU Training](#multi-gpu-training)
- [Key Features Explained](#key-features-explained)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [References & Resources](#references--resources)
- [License](#license)

---

## Overview

**TVERSKY-SIMILARITY-GRAD** is a modular and testable deep-learning codebase supporting research on **Tversky Projection Layers**—a newly proposed neural architecture for **psychologically plausible similarity** (Doumbouya et al., 2025).

This framework enables:
- ✅ Plug-and-play evaluation of different projection heads
- ✅ Simple model benchmarking on MNIST, Fruits-360, and other datasets
- ✅ Clean separation between **architecture**, **training**, and **research**
- ✅ Reproducible, testable experimentation
- ✅ Transfer learning with progressive unfreezing
- ✅ Mixed precision training (AMP)
- ✅ TensorBoard logging with Tversky parameter tracking

### What is Tversky Similarity?

Standard deep learning layers (e.g., dot-product linear layers) model **geometric similarity**, which fails to capture key properties of human perception—especially asymmetry ("the son resembles the father" ≠ "the father resembles the son").

The **Tversky contrast model** defines similarity in terms of:
- **Common features** (shared characteristics)
- **Distinctive features** (unique to each object)
- **Weighted asymmetry** (α and β parameters)

This repository provides the scaffolding to:
- Build CV pipelines using Tversky similarity
- Compare against baselines (linear, cosine)
- Experiment easily with new backbones + datasets

---

## Key Features

- 🚀 **Config-Driven Architecture**: Swap models via YAML configs, no code changes needed
- 🎯 **Transfer Learning**: ImageNet pretrained weights, progressive unfreezing strategies
- ⚡ **Optimized Training**: Mixed precision (AMP), automatic batch size selection, gradient clipping
- 📊 **TensorBoard Integration**: Automatic logging of metrics, weights, gradients, and Tversky parameters (alpha/beta)
- 🔄 **Parameter Sharing**: GlobalFeature bank for efficient parameter reduction
- 🧩 **Modular Design**: Composable backbones and heads via registry system
- 📓 **Notebook Support**: Ready-to-use Colab notebooks with full examples
- 🖥️ **Multi-GPU Support**: Distributed Data Parallel (DDP) training
- 🧪 **Testable**: Comprehensive unit tests for all components

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (see [PyTorch installation guide](https://pytorch.org/get-started/locally/))
- CUDA-capable GPU (optional, but recommended for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/tverskysimilaritygrad.git
cd tversky-similarity-grad
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality:

```bash
# TensorBoard for visualization
pip install tensorboard

# Jupyter for notebooks
pip install jupyter notebook

# Additional datasets
pip install datasets transformers
```

---

## Quick Start

### Example 1: Train on MNIST (5 minutes)

```bash
# Activate environment
source .venv/bin/activate

# Train with default config
python -m tverskycv.cli.train --config tverskycv/configs/mnist.yaml
```

Trained model checkpoints automatically save to `./checkpoints/`

### Example 2: Train on Fruits-360 (CPU-optimized)

```bash
# Use CPU-optimized config
python train_fruits360.py --config tverskycv/configs/fruits_cpu.yaml
```

### Example 3: Quick Test Run

```bash
# Create a minimal test config
cat > test_config.yaml << EOF
seed: 1337
dataset:
  name: mnist
  params:
    data_dir: ./data
    batch_size: 64
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
  epochs: 3
  lr: 1e-3
  device: cuda  # or "cpu"
EOF

# Run training
python -m tverskycv.cli.train --config test_config.yaml
```

---

## Usage Methods

### Command-Line Interface (CLI)

The CLI is the simplest way to train models using YAML configuration files.

#### Basic Training

```bash
python -m tverskycv.cli.train --config tverskycv/configs/mnist.yaml
```

#### Evaluation

```bash
python -m tverskycv.cli.eval \
    --config tverskycv/configs/mnist.yaml \
    --ckpt checkpoints/best_model.pt
```

#### Training Scripts

For more advanced training with logging and TensorBoard:

```bash
# Fruits-360 training with enhanced logging
python train_fruits360.py --config tverskycv/configs/fruits.yaml

# Optimized training with transfer learning
python train_fruits360_optimized.py --config fruits360_optimized_config.yaml
```

#### CLI Options

```bash
# Resume from checkpoint
python train_fruits360.py \
    --config tverskycv/configs/fruits.yaml \
    --ckpt checkpoints/latest.pt

# Custom log directory
python train_fruits360.py \
    --config tverskycv/configs/fruits.yaml \
    --log-dir ./logs/my_experiment

# Disable TensorBoard
python train_fruits360.py \
    --config tverskycv/configs/fruits.yaml \
    --no-tensorboard
```

### Python API

For programmatic control and custom training loops, use the Python API.

#### Basic Training with OptimizedTrainer

```python
import torch
from torch.utils.data import DataLoader
from tverskycv.training import OptimizedTrainer, create_optimized_dataloaders
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.registry import BACKBONES, HEADS

# Create model
backbone = BACKBONES.get('resnet18')(out_dim=128, pretrained=True)
head = HEADS.get('linear')(in_dim=128, num_classes=10)
model = ImageClassifier(backbone, head)

# Create data loaders
train_loader, val_loader = create_optimized_dataloaders(
    train_dataset, val_dataset
)

# Initialize trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = OptimizedTrainer(
    model=model,
    device=device,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir='checkpoints',
    use_tensorboard=True  # Default: True
)

# Train
results = trainer.train(train_loader, val_loader)
print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
```

#### Transfer Learning with Progressive Unfreezing

```python
from tverskycv.training import OptimizedTrainer

# Define progressive unfreezing schedule
schedule = {
    0: 0.0,      # Epochs 0-4: Freeze all backbone (train head only)
    5: 0.33,     # Epochs 5-9: Unfreeze top 33% of backbone
    10: 0.66,    # Epochs 10-14: Unfreeze top 66% of backbone
    15: 1.0      # Epochs 15+: Unfreeze all layers
}

trainer = OptimizedTrainer(
    model=model,
    device=device,
    num_epochs=30,
    progressive_unfreezing=schedule,  # Enable progressive unfreezing
    checkpoint_dir='checkpoints'
)

results = trainer.train(train_loader, val_loader)
```

#### Static Backbone Freezing

```python
trainer = OptimizedTrainer(
    model=model,
    device=device,
    num_epochs=10,
    freeze_backbone=True,  # Freeze backbone for entire training
    checkpoint_dir='checkpoints'
)
```

#### Manual Transfer Learning Control

```python
from tverskycv.training import (
    freeze_backbone,
    get_trainable_params,
    ProgressiveUnfreezing
)

# Freeze backbone manually
freeze_backbone(model, freeze=True)

# Check trainable parameters
trainable = get_trainable_params(model)
total = get_total_params(model)
print(f"Trainable: {trainable:,} / {total:,}")

# Use ProgressiveUnfreezing manager
unfreezer = ProgressiveUnfreezing(model, schedule)
for epoch in range(num_epochs):
    ratio = unfreezer.unfreeze_for_epoch(epoch)
    info = unfreezer.get_trainable_info()
    print(f"Epoch {epoch}: {info['trainable']:,} trainable params")
```

#### Loading Models and TensorBoard Logs

The `load_model()` function provides a convenient way to load trained models along with their associated TensorBoard logs and training metadata.

**Basic Usage:**

```python
from tverskycv import load_model

# Load model with automatic TensorBoard log discovery
result = load_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
checkpoint_info = result['checkpoint_info']
tensorboard_log_dir = result['tensorboard_log_dir']

# Access training metadata
print(f"Best validation accuracy: {checkpoint_info.get('best_val_acc', 'N/A'):.4f}")
print(f"Trained for {checkpoint_info.get('epoch', 'N/A')} epochs")
print(f"TensorBoard logs: {tensorboard_log_dir}")

# View TensorBoard (in Jupyter/Colab)
# %tensorboard --logdir {tensorboard_log_dir}
```

**Loading with Optimizer State:**

```python
# Load model and optimizer state for resuming training
result = load_model(
    checkpoint_path='checkpoints/checkpoint_epoch_10.pt',
    config_path='config.yaml',
    load_optimizer=True
)

model = result['model']
optimizer_state = result['optimizer_state_dict']
scheduler_state = result['scheduler_state_dict']

# Restore optimizer state if needed
if optimizer_state:
    optimizer.load_state_dict(optimizer_state)
if scheduler_state:
    scheduler.load_state_dict(scheduler_state)
```

**Listing Available Checkpoints:**

```python
from tverskycv import list_checkpoints

# List all checkpoints in a directory
checkpoints = list_checkpoints('checkpoints/')

# Checkpoints are sorted by best validation accuracy
for ckpt in checkpoints:
    print(f"Path: {ckpt['path']}")
    print(f"  Best Val Acc: {ckpt.get('best_val_acc', 'N/A'):.4f}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Size: {ckpt['size_mb']:.2f} MB")
    print()

# Find the best checkpoint
best_ckpt = checkpoints[0]  # Already sorted by best_val_acc
print(f"Best checkpoint: {best_ckpt['path']}")
```

**Finding TensorBoard Logs Manually:**

```python
from tverskycv.training.checkpoint import find_tensorboard_logs

# Find TensorBoard logs for a specific checkpoint
tb_log_dir = find_tensorboard_logs('checkpoints/best_model.pt')

if tb_log_dir:
    print(f"TensorBoard logs found: {tb_log_dir}")
    # Start TensorBoard: tensorboard --logdir {tb_log_dir}
else:
    print("TensorBoard logs not found")
```

**Complete Example: Load and Evaluate:**

```python
from tverskycv import load_model, evaluate

# Load trained model
result = load_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
model.eval()

# Evaluate on validation set
eval_results = evaluate(model, 'tverskycv/configs/fruits.yaml', split='val')
print(f"Validation accuracy: {eval_results['accuracy']:.4f}")

# View training history in TensorBoard
if result['tensorboard_log_dir']:
    print(f"\nTo view training history:")
    print(f"  tensorboard --logdir {result['tensorboard_log_dir']}")
```

### Jupyter Notebooks

Interactive notebooks are available for experimentation and learning.

#### Available Notebooks

1. **`Classification_Colab.ipynb`**: Complete Fruits-360 classification example
   - Transfer learning with pretrained ResNet18
   - TverskyReduceBackbone with parameter sharing
   - TensorBoard integration
   - Alpha/beta parameter tracking
   - Model comparison and visualization

2. **`TverskySimilarity.ipynb`**: Tversky similarity concepts and examples

3. **`TverskyGPT_Colab.ipynb`**: Tversky-based GPT model for NLP

#### Using in Google Colab

1. Open the notebook in Colab
2. Run the setup cell to clone the repository
3. Follow the cells sequentially
4. All dependencies are installed automatically

#### Using Locally

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab

# Navigate to tverskycv/notebooks/ and open the desired notebook
```

### Loading Models and Checkpoints

After training, you can easily load models along with their TensorBoard logs and training metadata.

**Quick Example:**

```python
from tverskycv import load_model

# Load model with automatic TensorBoard discovery
result = load_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
print(f"Best accuracy: {result['checkpoint_info']['best_val_acc']:.4f}")
print(f"TensorBoard logs: {result['tensorboard_log_dir']}")
```

**Key Features:**
- ✅ Automatic TensorBoard log discovery
- ✅ Checkpoint metadata extraction (epoch, accuracy, etc.)
- ✅ Config-aware model rebuilding
- ✅ Optional optimizer/scheduler state loading
- ✅ List and compare checkpoints

See the [Checkpoint Management](#checkpoint-management) section for detailed examples.

### Loading Models and Checkpoints

After training, you can easily load models along with their TensorBoard logs and training metadata.

**Quick Example:**

```python
from tverskycv import load_model

# Load model with automatic TensorBoard discovery
result = load_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
print(f"Best accuracy: {result['checkpoint_info']['best_val_acc']:.4f}")
print(f"TensorBoard logs: {result['tensorboard_log_dir']}")
```

**Key Features:**
- ✅ Automatic TensorBoard log discovery
- ✅ Checkpoint metadata extraction (epoch, accuracy, etc.)
- ✅ Config-aware model rebuilding
- ✅ Optional optimizer/scheduler state loading
- ✅ List and compare checkpoints

See the [Checkpoint Management](#checkpoint-management) section for detailed examples.

### Multi-GPU Training

For faster training on multiple GPUs, use Distributed Data Parallel (DDP).

#### Single GPU (OptimizedTrainer)

```python
from tverskycv.training import OptimizedTrainer

trainer = OptimizedTrainer(
    model=model,
    device=torch.device('cuda:0'),
    num_epochs=10
)
results = trainer.train(train_loader, val_loader)
```

#### Multi-GPU (DDP)

```bash
# Train on 4 GPUs
python train_multi_gpu.py --gpus 4 --batch-size 256 --epochs 10
```

#### Using the Launcher

```python
from tverskycv.training import launch_distributed_training

launch_distributed_training(
    num_gpus=4,
    config_path='tverskycv/configs/fruits.yaml'
)
```

#### Performance Expectations

| Platform | GPUs | Time (10 epochs, Fruits-360) | Cost |
|----------|------|------------------------------|------|
| Colab Free (T4) | 1 | 30-40 min | FREE |
| GCP (A100) | 1 | 10-15 min | ~$0.90 |
| GCP (A100) | 4 | 3-5 min | ~$1.20 |
| Lambda (RTX 6000) | 4 | 3-5 min | ~$0.37 |

---

## Key Features Explained

### Transfer Learning

Transfer learning uses pretrained ImageNet weights to improve performance and convergence speed.

**Features:**
- ImageNet pretrained ResNet backbones
- Progressive unfreezing strategies
- Static backbone freezing option
- Automatic parameter tracking

**Example:**
```python
# Load pretrained ResNet18
backbone = BACKBONES.get('resnet18')(pretrained=True, out_dim=128)

# Use in model
model = ImageClassifier(backbone, head)
```

### Mixed Precision Training (AMP)

Automatic Mixed Precision (AMP) uses FP16 for faster training with minimal accuracy impact.

**Benefits:**
- ~2x faster training
- Lower GPU memory usage
- Automatic precision management

**Usage:**
```python
# Automatically enabled in OptimizedTrainer when CUDA is available
trainer = OptimizedTrainer(model=model, device=torch.device('cuda'))
```

### TensorBoard Logging

Comprehensive logging to TensorBoard for visualization and monitoring.

**Logged Metrics:**
- Training/validation loss and accuracy
- Learning rate schedule
- Model weights and gradients (periodic)
- **Tversky parameters (alpha/beta)** - automatically tracked
- Transfer learning metrics (unfreeze ratio, trainable params)

**Usage:**
```bash
# Start TensorBoard
tensorboard --logdir checkpoints/tensorboard

# Or for training scripts
tensorboard --logdir logs
```

**View in Browser:**
Open `http://localhost:6006` to see:
- Loss curves
- Accuracy plots
- Tversky parameter evolution (alpha/beta)
- Weight distributions
- Learning rate schedule

### Config-Driven Architecture

Swap models, datasets, and training settings via YAML configs without code changes.

**Example:**
```yaml
model:
  backbone:
    name: resnet18  # Change to 'simple_cnn' or 'tversky_reduce_compact'
    params:
      pretrained: true
      out_dim: 128
  head:
    name: linear  # Change to 'tversky'
    params:
      in_dim: 128
      num_classes: 10
```

### Model Registry System

Register and use custom backbones and heads via the registry.

```python
from tverskycv.registry import BACKBONES, HEADS

# Register custom backbone
@BACKBONES.register("my_backbone")
def build_my_backbone(out_dim: int = 128, **_):
    return MyCustomBackbone(out_dim=out_dim)

# Use in config
# model:
#   backbone:
#     name: my_backbone
```

### Parameter Sharing (GlobalFeature Bank)

Share feature matrices and Tversky parameters across layers to reduce parameter count.

**Benefits:**
- Significant parameter reduction (50%+ in some cases)
- Memory efficient
- Maintains model performance

**Usage:**
```python
from tverskycv.models.backbones.tversky_reduce_backbone import SharedTverskyCompact

layer = SharedTverskyCompact(
    in_features=512,
    n_prototypes=128,
    feature_key='shared',  # Same key = shared parameters
    share_params=True
)
```

---

## Configuration

Configuration files are YAML-based and control all aspects of training.

### Basic Config Structure

```yaml
seed: 1337

dataset:
  name: mnist  # or 'fruits_360'
  params:
    data_dir: ./data
    batch_size: 256
    num_workers: 4

model:
  backbone:
    name: simple_cnn  # or 'resnet18', 'tversky_reduce_compact'
    params:
      out_dim: 128
      pretrained: false  # Set to true for transfer learning
  head:
    name: linear  # or 'tversky'
    params:
      in_dim: 128
      num_classes: 10

train:
  epochs: 10
  lr: 1e-3
  weight_decay: 1e-4
  device: cuda  # or 'cpu'
  ckpt_dir: ./checkpoints

logging:
  use_tensorboard: true
  use_wandb: false
```

### Available Config Files

- `tverskycv/configs/mnist.yaml` - MNIST classification
- `tverskycv/configs/fruits.yaml` - Fruits-360 (GPU)
- `tverskycv/configs/fruits_cpu.yaml` - Fruits-360 (CPU-optimized)
- `tverskycv/configs/models/resnet18.yaml` - ResNet18 backbone config
- `tverskycv/configs/models/tversky_reduce_compact.yaml` - TverskyReduceBackbone config

### Modifying Configs

Simply edit the YAML file or create a new one:

```bash
# Copy existing config
cp tverskycv/configs/mnist.yaml my_config.yaml

# Edit my_config.yaml
# Then use it:
python -m tverskycv.cli.train --config my_config.yaml
```

---

## Model Architectures

### Available Backbones

#### SimpleCNN
Basic convolutional neural network for quick experiments.

```yaml
backbone:
  name: simple_cnn
  params:
    out_dim: 128
```

#### ResNet18
ResNet18 with optional ImageNet pretrained weights.

```yaml
backbone:
  name: resnet18
  params:
    out_dim: 128
    pretrained: true  # Use ImageNet weights
    in_channels: 3
```

#### TverskyReduceBackbone
CNN with Tversky projection layers and parameter sharing.

```yaml
backbone:
  name: tversky_reduce_compact
  params:
    out_dim: 128
    in_channels: 3
    img_size: 64
    feature_key: main
    share_features: true
    alpha: 1.0
    beta: 1.0
```

### Available Heads

#### LinearHead
Standard linear classification head.

```yaml
head:
  name: linear
  params:
    in_dim: 128
    num_classes: 10
```

#### TverskyHead
Tversky-based projection head.

```yaml
head:
  name: tversky
  params:
    in_dim: 128
    num_classes: 10
```

### Swapping Models

Change models by editing the config file—no code changes needed:

```yaml
# Switch from SimpleCNN to ResNet18
model:
  backbone:
    name: resnet18  # Changed from 'simple_cnn'
    params:
      pretrained: true  # Added pretrained weights
      out_dim: 128
```

---

## Advanced Features

### Transfer Learning Utilities

```python
from tverskycv.training import (
    freeze_backbone,
    get_trainable_params,
    ProgressiveUnfreezing
)

# Freeze/unfreeze backbone
freeze_backbone(model, freeze=True)

# Get parameter counts
trainable = get_trainable_params(model)
total = get_total_params(model)

# Progressive unfreezing
unfreezer = ProgressiveUnfreezing(model, schedule={0: 0.0, 5: 0.33, 10: 0.66, 15: 1.0})
```

### Checkpoint Management

#### Using the High-Level API (Recommended)

The `load_model()` function is the easiest way to load models with automatic TensorBoard log discovery:

```python
from tverskycv import load_model, list_checkpoints

# Load model with TensorBoard logs
result = load_model(
    checkpoint_path='checkpoints/best_model.pt',
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
print(f"Best accuracy: {result['checkpoint_info']['best_val_acc']:.4f}")
print(f"TensorBoard: {result['tensorboard_log_dir']}")

# List all checkpoints
checkpoints = list_checkpoints('checkpoints/')
for ckpt in checkpoints:
    print(f"{ckpt['path']}: {ckpt.get('best_val_acc', 'N/A')}")
```

#### Using Low-Level Utilities

For more control, use the low-level checkpoint utilities:

```python
from tverskycv.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_checkpoint_info,
    find_tensorboard_logs
)

# Save checkpoint
save_checkpoint(
    'checkpoints/model.pt',
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    val_acc=val_acc,
    best_val_acc=best_val_acc
)

# Load checkpoint (requires model instance)
checkpoint = load_checkpoint(
    'checkpoints/model.pt',
    model=model,
    device=device,
    optimizer=optimizer
)

# Get checkpoint metadata without loading model
info = get_checkpoint_info('checkpoints/best_model.pt')
print(f"Epoch: {info.get('epoch')}, Best Acc: {info.get('best_val_acc')}")

# Find TensorBoard logs
tb_dir = find_tensorboard_logs('checkpoints/best_model.pt')
if tb_dir:
    print(f"TensorBoard logs: {tb_dir}")
```

#### Checkpoint Directory Structure

When using `OptimizedTrainer`, checkpoints are organized as:

```
checkpoints/
├── best_model.pt          # Best model by validation accuracy
├── checkpoint_epoch_5.pt  # Periodic checkpoints
├── checkpoint_epoch_10.pt
└── tensorboard/          # TensorBoard event files
    └── events.out.tfevents.*
```

The `load_model()` function automatically searches for TensorBoard logs in common locations:
- `checkpoint_dir/tensorboard/`
- `checkpoint_dir/tb_logs/`
- `checkpoint_dir/logs/`
- `checkpoint_dir/runs/`

### ONNX Export

Export models to ONNX format for deployment:

```bash
python -m tverskycv.scripts.export_onnx \
    --config tverskycv/configs/mnist.yaml \
    --ckpt checkpoints/best_model.pt \
    --out model.onnx
```

### Custom Model Registration

Register custom models for use in configs:

```python
from tverskycv.registry import BACKBONES, HEADS

@BACKBONES.register("my_custom_backbone")
def build_my_backbone(out_dim: int = 128, **kwargs):
    return MyCustomBackbone(out_dim=out_dim, **kwargs)

# Now use in config:
# backbone:
#   name: my_custom_backbone
```

---

## Examples

### Example 1: MNIST Classification

```bash
# Train on MNIST
python -m tverskycv.cli.train --config tverskycv/configs/mnist.yaml

# Evaluate
python -m tverskycv.cli.eval \
    --config tverskycv/configs/mnist.yaml \
    --ckpt checkpoints/best_model.pt
```

### Example 2: Fruits-360 with Transfer Learning

```python
import torch
from tverskycv.registry import BACKBONES, HEADS
from tverskycv.models.wrappers.classifiers import ImageClassifier
from tverskycv.training import OptimizedTrainer

# Create model with pretrained ResNet18
backbone = BACKBONES.get('resnet18')(
    pretrained=True,
    out_dim=128
)
head = HEADS.get('linear')(in_dim=128, num_classes=113)
model = ImageClassifier(backbone, head)

# Setup trainer with progressive unfreezing
schedule = {0: 0.0, 5: 0.33, 10: 0.66, 15: 1.0}
trainer = OptimizedTrainer(
    model=model,
    device=torch.device('cuda'),
    num_epochs=30,
    progressive_unfreezing=schedule
)

# Train (assuming data loaders are set up)
results = trainer.train(train_loader, val_loader)
```

### Example 3: Multi-GPU Training

```bash
# Train on 4 GPUs
python train_multi_gpu.py \
    --gpus 4 \
    --batch-size 256 \
    --epochs 10 \
    --config tverskycv/configs/fruits.yaml
```

### Example 4: Loading and Evaluating a Trained Model

```python
from tverskycv import load_model, evaluate, list_checkpoints

# List available checkpoints
checkpoints = list_checkpoints('checkpoints/')
print(f"Found {len(checkpoints)} checkpoints")

# Load the best model
best_ckpt = checkpoints[0]  # Sorted by best_val_acc
result = load_model(
    checkpoint_path=best_ckpt['path'],
    config_path='tverskycv/configs/fruits.yaml'
)

model = result['model']
print(f"Loaded model from epoch {result['checkpoint_info'].get('epoch')}")
print(f"Best validation accuracy: {result['checkpoint_info'].get('best_val_acc'):.4f}")

# Evaluate on validation set
eval_results = evaluate(model, 'tverskycv/configs/fruits.yaml', split='val')
print(f"Current validation accuracy: {eval_results['accuracy']:.4f}")

# View TensorBoard logs
if result['tensorboard_log_dir']:
    print(f"\nView training history:")
    print(f"  tensorboard --logdir {result['tensorboard_log_dir']}")
```

### Example 5: Custom Training Loop

```python
from tverskycv.training.engine import train_one_epoch, evaluate
from tverskycv.training.utils import set_seed, resolve_device

set_seed(42)
device = resolve_device('cuda')

# Custom training loop
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_metrics = evaluate(model, val_loader, device, criterion)
    print(f"Epoch {epoch}: Train Acc={train_metrics['accuracy']:.4f}, "
          f"Val Acc={val_metrics['accuracy']:.4f}")
```

---

## Troubleshooting

### Common Issues

#### Out of Memory

**Solution:**
- Reduce batch size in config: `batch_size: 8` (or 4)
- Reduce image size: `img_size: 32`
- Use CPU if GPU memory is limited

#### Slow Training

**Solutions:**
- Enable mixed precision (automatic in OptimizedTrainer)
- Reduce `num_workers` for CPU training
- Use smaller image size
- Enable GPU if available

#### Import Errors

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### TensorBoard Not Found

**Solution:**
```bash
pip install tensorboard
```

#### Model Not Learning

**Check:**
- Learning rate (try 1e-4 to 1e-2)
- Data normalization matches pretrained model (ImageNet stats)
- Labels are correct (0 to num_classes-1)
- Enable pretrained weights for transfer learning

### Performance Tips

1. **Use Transfer Learning**: Always use `pretrained: true` for ResNet backbones
2. **Progressive Unfreezing**: Start with frozen backbone, gradually unfreeze
3. **Mixed Precision**: Automatically enabled in OptimizedTrainer
4. **Batch Size**: Larger batches = more stable training (if memory allows)
5. **Learning Rate**: Use learning rate scheduling (OneCycleLR in OptimizedTrainer)

### Hardware-Specific Notes

**CPU Training:**
- Use `fruits_cpu.yaml` config
- Reduce batch size to 8-16
- Set `num_workers: 2-4` based on CPU cores
- Expect longer training times

**GPU Training:**
- Use default configs
- Batch size 32-64 typically works
- Enable mixed precision for 2x speedup
- Monitor GPU memory usage

---

## References & Resources

### Documentation Files

- [`CLI_TRAINING_EXAMPLES.md`](CLI_TRAINING_EXAMPLES.md) - Detailed CLI usage examples
- [`QUICK_START_TRAINING.md`](QUICK_START_TRAINING.md) - Quick start guide
- [`README_TRAINING.md`](README_TRAINING.md) - Multi-GPU training guide

### Research Papers

- **Doumbouya et al. (2025)**.  
  *Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity*  
  https://arxiv.org/abs/2506.11035

- **Tversky, A. (1977)**.  
  *Features of similarity*. Psychological Review.

### Related Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Hugging Face Datasets](https://huggingface.co/datasets)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Acknowledgments

This framework is based on research into Tversky similarity for neural networks. Special thanks to the research community for advancing psychologically plausible deep learning.

---

**Happy Training! 🚀**
