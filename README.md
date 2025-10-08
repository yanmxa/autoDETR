# DETR: DEtection TRansformer

End-to-end object detection using Transformer encoder-decoder with ResNet50 backbone.

## Project Structure

```
detr/
├── src/detr/                    # Main package
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── detr.py             # DETR model implementation
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset class
│   │   └── transforms.py        # Image transforms
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging setup
│   │   └── metrics.py           # Evaluation metrics
│   ├── tools/                   # Tools and scripts
│   │   ├── __init__.py
│   │   └── collect_images.py    # Image collection tool
│   ├── __init__.py
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   └── inference.py             # Inference script
├── data/                        # Data directory
│   ├── images/                  # Training images (organized by class)
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── configs/                     # Configuration files
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── pyproject.toml              # Project configuration
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) - an extremely fast Python package installer written in Rust.

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 1. Create Virtual Environment with uv

```bash
# Create virtual environment using uv (much faster than python -m venv)
uv venv

# This creates a .venv directory automatically
```

### 2. Install Dependencies with uv

uv automatically detects and uses the `.venv` environment - no need to activate!

```bash
# Install project and all dependencies
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Verify installation
uv pip list | grep -E "torch|opencv|rich"
```

**Why use uv?**
- 10-100x faster than pip
- Automatic virtual environment detection
- Better dependency resolution
- Written in Rust for performance

## Workflow

### 1. Collect Training Images

Use the built-in image collection tool to capture training data from your webcam.

```bash
# Run image collection script
python src/detr/tools/collect_images.py

# Or use the installed command
uv run detr-collect
```

**Configuration:**
- Edit class names in `src/detr/tools/collect_images.py`
- Default: 30 images per class
- Images saved to `data/images/<class_name>/`

**Tips:**
- Press `q` to quit early
- Ensure good lighting
- Vary hand positions and angles

### 2. Label Data with Label Studio

Use Label Studio for bounding box annotation and quality control.

```bash
# Install Label Studio
uv pip install label-studio

# Verify installation
uv pip list | grep label-studio

# Launch Label Studio
uv run label-studio
```

**Label Studio Setup:**
1. Open browser at http://localhost:8080
2. Create new project for **Object Detection with Bounding Boxes**
3. Import images from `data/images/`
4. Configure labeling interface (select bounding box tool)
5. Start labeling - draw boxes around objects

**Keyboard Shortcuts:**
- `Ctrl + Enter` - Submit label
- `1`, `2`, `3` - Quick label selection
- `Ctrl + Z` - Undo
- `Delete` - Remove bounding box

**Best Practices:**
- Draw tight bounding boxes around objects
- Label all objects in each image
- Ensure consistent labeling across images
- Export annotations in COCO format to `data/processed/`

### 3. Train Model

Train the DETR model on your labeled data.

```bash
# Basic training with uv
uv run python src/detr/train.py --data-dir data/images --epochs 50 --batch-size 32

# With custom parameters
uv run python src/detr/train.py \
    --data-dir data/images \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --device cuda

# Or use the installed command
uv run detr-train --data-dir data/images --epochs 50
```

**Training Tips:**
- Monitor training with TensorBoard: `tensorboard --logdir logs/tensorboard`
- Best model saved to `checkpoints/best_model.pth`
- Adjust batch size based on GPU memory
- Use `--device cpu` if no GPU available

### 4. Evaluate Model

Evaluate model performance on test data.

```bash
# Evaluate with uv
uv run python src/detr/eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/images

# Or use the installed command
uv run detr-eval --checkpoint checkpoints/best_model.pth --data-dir data/images
```

### 5. Inference

Run inference on new images or webcam.

```bash
# On a single image
uv run python src/detr/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg

# Real-time webcam inference
uv run python src/detr/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --webcam --camera-id 0

# Or use the installed command
uv run detr-infer --checkpoint checkpoints/best_model.pth --webcam
```

## Model Architecture

DETR combines:
- **ResNet-50** (pretrained on ImageNet) as the backbone for feature extraction
- **Transformer Encoder** for processing image features
- **Transformer Decoder** with learned object queries
- **Prediction Heads** for class labels and bounding boxes

Key features:
- 2D sine-cosine positional encoding for spatial awareness
- Multi-head self-attention and cross-attention
- End-to-end training without NMS (Non-Maximum Suppression)
- Bipartite matching for set prediction

**Model Output:**
- `pred_logits`: Class predictions (batch, num_queries, num_classes + 1)
- `pred_boxes`: Bounding boxes in (cx, cy, w, h) format, normalized to [0, 1]

## Configuration

Edit training parameters in `configs/train_config.yaml`:

```yaml
# configs/train_config.yaml
model:
  hidden_dim: 256
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  num_queries: 100

training:
  epochs: 300
  batch_size: 2
  learning_rate: 1e-4
  weight_decay: 1e-4

data:
  image_size: 800
  train_split: 0.8
```

## Test the Model

```bash
# Test model forward pass
python src/detr/models/detr.py
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black src/

# Type checking
uv run mypy src/

# Lint
uv run flake8 src/
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- Rich >= 13.7.0

See [pyproject.toml](pyproject.toml) for full dependencies.

## Citation

If you find this code useful, please cite:

```bibtex
@misc{detr2024,
  title={DETR: DEtection TRansformer},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

