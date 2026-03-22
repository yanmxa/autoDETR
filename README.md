# autoDETR

Automated experimentation platform for DETR (DEtection TRansformer) object detection. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Train, validate, iterate. Each experiment produces metrics and a visual snapshot, so you can track exactly how changes affect detection quality.

## Quick Start

```bash
# Install
pip install -e .

# 1. Run baseline training
autodetr-train

# 2. Validate and generate visualization
autodetr-val --tag baseline

# 3. Plot experiment history
autodetr-plot
```

## Experiment Workflow

See [program.md](program.md) for the full experiment protocol.

```
Edit config.py/train.py
    |
    v
git commit -s
    |
    v
autodetr-train > run.log 2>&1
    |
    v
autodetr-val --tag <name>     -->  val_results/val_<name>.png
    |
    v
Record in results.tsv
    |
    v
Keep (advance) or Discard (git reset)
```

Each validation run produces:
- **Metrics**: val_loss, accuracy, mean_iou, num_detections (grep-friendly format)
- **Visualization**: 2x5 grid of test images with GT boxes (blue) and predictions (green)
- **History plot**: `autodetr-plot` charts accuracy/loss/IoU across all experiments

## Commands

| Command | Description |
|---------|-------------|
| `autodetr-train` | Train model, auto-runs validation at end |
| `autodetr-val` | Run validation on test set, save visualization |
| `autodetr-eval` | Interactive evaluation with visualization |
| `autodetr-plot` | Plot experiment history from results.tsv |
| `autodetr-collect` | Capture training images from webcam |

## Project Structure

```
autoDETR/
├── program.md              # Experiment protocol
├── results.tsv             # Experiment log (not tracked by git)
├── val_results/            # Validation visualizations (tracked by git)
│   ├── val_baseline.png
│   └── val_<tag>.png
├── src/detr/
│   ├── config.py           # Centralized configuration (editable)
│   ├── train.py            # Training loop (editable)
│   ├── validate.py         # Validation harness (read-only)
│   ├── evaluate.py         # Interactive evaluation
│   ├── model/              # DETR architecture (read-only)
│   ├── loss/               # Loss functions (read-only)
│   ├── data/               # Dataset loader (read-only)
│   ├── utils/              # Display utilities
│   └── tools/              # Data collection & plotting
├── data/
│   ├── raw/                # Original captured images
│   ├── process/            # Labeled dataset (images + labels)
│   ├── train/              # Training split (170 samples)
│   └── test/               # Test split (32 samples)
└── checkpoints/            # Model weights
```

## Validation Output

```
---
val_loss:         2.345678
accuracy:         0.7500
mean_iou:         0.6234
num_detections:   8
confidence_threshold: 0.10
experiment_tag:   baseline
val_image:        val_results/val_baseline.png
---
```

## Current Baseline

| Parameter | Value |
|-----------|-------|
| Classes | 3 (one, two, three) |
| Encoder/Decoder layers | 1+1 |
| Object queries | 25 |
| Hidden dim | 256 |
| Epochs | 150 |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Optimizer | Adam |
| Training samples | 170 |

## Requirements

- Python >= 3.12
- PyTorch >= 2.0.0
- torchvision, albumentations, rich, matplotlib

## License

MIT License
