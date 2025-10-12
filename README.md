# DETR: DEtection TRansformer

End-to-end object detection using Transformer encoder-decoder with ResNet50 backbone.

## Quick Start

```bash
# Install
pip install -e .

# 1. Collect images
detr-collect

# 2. Train model
detr-train

# 3. Evaluate model
detr-eval
```

## Features

- **End-to-end object detection** with Transformer architecture
- **ResNet-50 backbone** (ImageNet pretrained)
- **Hungarian matching** for optimal assignment
- **Rich console output** with beautiful progress bars
- **Automatic loss curve plotting** after training
- **Data reorganization tool** for balanced class distribution

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

## Basic Workflow

### 1. Collect Images

```bash
detr-collect
```

Captures training images from webcam (30 images per class, 2s interval).

### 2. Train Model

```bash
detr-train
```

Trains DETR model with:

- 100 epochs (configurable)
- Automatic checkpointing
- Loss curve visualization saved to `training_curves.png`

### 3. Evaluate Model

```bash
detr-eval
```

Evaluates model on test set and saves visualization to `evaluation_results.png`.

## Current Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Encoder layers | 2 | Transformer encoder depth |
| Decoder layers | 2 | Transformer decoder depth |
| Object queries | 100 | Max objects per image |
| Epochs | 100 | Training iterations |
| Learning rate | 1e-4 | Adam optimizer LR |
| Batch size | 4 | Images per batch |

**To modify**: Edit `get_training_config()` in [src/detr/train.py](src/detr/train.py)

## Model Architecture

```text
Input Image (224×224)
    ↓
ResNet-50 Backbone
    ↓
Transformer Encoder (2 layers)
    ↓
Transformer Decoder (2 layers) ← 100 Object Queries
    ↓
Prediction Heads
    ├─ Classification (num_classes + 1)
    └─ Bounding Boxes (cx, cy, w, h)
```

## Loss Function

DETR combines three losses with Hungarian matching:

1. **Classification Loss** (weight: 2.0) - Cross-entropy for object classes
2. **BBox L1 Loss** (weight: 10.0) - L1 distance for box coordinates
3. **GIoU Loss** (weight: 5.0) - Generalized IoU for box overlap

## Command-Line Tools

| Command | Description |
|---------|-------------|
| `detr-collect` | Capture training images from webcam |
| `detr-train` | Train DETR model with loss tracking |
| `detr-eval` | Evaluate trained model on test set |

## Project Structure

```text
DETR/
├── src/detr/
│   ├── data/              # Dataset loader
│   ├── model/             # DETR implementation
│   ├── loss/              # Loss functions & matcher
│   ├── utils/             # Display utilities
│   ├── tools/             # Collection & reorganization
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── data_new/              # Training/test data
├── checkpoints/           # Saved models
├── training_curves.png    # Loss visualization
└── evaluation_results.png # Results visualization
```

## Tips for Better Results

### Data Quality

- Collect 100+ diverse images per class
- Ensure good lighting and varied angles
- Use tight bounding boxes around objects
- Balance class distribution

### Training

- Monitor loss curves for overfitting/underfitting
- Start with current config (2+2 layers) for good balance
- Use GPU for 10-20x faster training
- Adjust confidence threshold in evaluation (0.5-0.9)

### Troubleshooting

| Problem | Solution |
|---------|----------|
| **Overfitting** (test loss ↑ while train loss ↓) | Increase dropout or collect more data |
| **Underfitting** (both losses stay high) | Increase model layers or lower dropout |
| **Slow convergence** | Increase learning rate (1e-4 → 1e-3) |
| **No detections** | Lower confidence threshold (0.7 → 0.3) |
| **CUDA OOM** | Reduce batch_size (4 → 2) |

## Documentation

- **[DIAGNOSIS.md](DIAGNOSIS.md)** - Complete optimization history & roadmap
- **[docs/](docs/)** - Detailed phase-by-phase optimization reports

## Requirements

- Python >= 3.12
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- albumentations >= 1.3.0
- rich >= 13.7.0

See [pyproject.toml](pyproject.toml) for full dependencies.

## Citation

Original DETR paper:

```bibtex
@inproceedings{carion2020end,
  title={End-to-end object detection with transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel
          and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European conference on computer vision},
  pages={213--229},
  year={2020},
  organization={Springer}
}
```

## License

MIT License
