# DETR Model

Implementation of DETR (Detection Transformer) with detailed parameter explanations.

## Overview

DETR is an end-to-end object detection model that uses Transformers instead of hand-crafted components like anchors and NMS.

```python
from detr.model import DETR

model = DETR(
    num_classes=91,
    num_queries=100,
    num_encoder_layers=6,
    num_decoder_layers=6,
    nheads=8
)
```

## Architecture

```
Input Image [B, 3, H, W]
        ↓
ResNet-50 Backbone
        ↓
Conv 2048→256
        ↓
Flatten + Positional Encoding
        ↓
┌─────────────────────────────────────────────────────┐
│ Transformer Encoder                                 │
│ (num_encoder_layers layers)                         │
│ Each layer has nheads heads                         │
│                                                     │
│ Input: Image features [B, 49, 256]                  │
│ Process: Self-attention on spatial locations        │
│ Output: Enhanced features [B, 49, 256]              │
└─────────────────────────────────────────────────────┘
        ↓
Enhanced Image Features [B, 49, 256]
        ↓
┌─────────────────────────────────────────────────────┐
│ Transformer Decoder                                 │
│ (num_decoder_layers layers)                         │
│ Each layer has nheads heads                         │
│                                                     │
│ Inputs:                                             │
│  1. Object Queries [B, num_queries, 256] ←──────┐  │
│     └─ Learnable embeddings (parameters)        │  │
│     └─ Each query = potential object detector   │  │
│     └─ Initialized randomly, learned end-to-end │  │
│                                                  │  │
│  2. Image Features [B, 49, 256]                 │  │
│     └─ From encoder                             │  │
│                                                  │  │
│ Process:                                         │  │
│  • Self-attention: Queries interact with each   │  │
│    other (detect different objects)             │  │
│  • Cross-attention: Queries attend to image     │  │
│    features (localize objects)                  │  │
│                                                  │  │
│ Output: Refined queries [B, num_queries, 256]   │  │
└─────────────────────────────────────────────────────┘
        ↓
Refined Query Embeddings [B, num_queries, 256]
        ↓
        ├─────────────────────────────────────┐
        ↓                                     ↓
Classification Head                    BBox Head
        ↓                                     ↓
[B, num_queries, num_classes+1]    [B, num_queries, 4]
   Each query predicts:              Each query predicts:
   - Class probabilities            - Box (cx, cy, w, h)
   - Including "no object"          - Normalized [0, 1]
```

**Key Concept: Object Queries**

Object queries are **learnable parameters** (not derived from the image):
- Think of them as "detection slots" or "object hypotheses"
- During training, each query learns to specialize in detecting certain types of objects
- Example with 100 queries:
  - Query 0 might learn to detect cars
  - Query 1 might learn to detect people on the left
  - Query 2 might learn to detect small objects
  - Query 99 might learn to detect objects in the background
- Queries that don't find objects predict "no object" class
- Unlike anchors (fixed positions), queries learn **what** to detect and **where** to look

## Key Parameters

### 1. `num_encoder_layers` / `num_decoder_layers`

**Purpose**: Control the reasoning depth of the model.

**Independence**: These two parameters are **completely independent**. You can set different values.

**Effect**:
- More layers = Better accuracy but slower
- Each layer refines the understanding

**Common Configurations**:

```python
# Symmetric (Standard)
num_encoder_layers=6, num_decoder_layers=6
# Balanced performance, DETR paper default

# Encoder-Heavy (Asymmetric)
num_encoder_layers=8, num_decoder_layers=4
# Better feature extraction, lighter decoding
# Use when: Complex scenes requiring deep understanding

# Decoder-Heavy (Asymmetric)
num_encoder_layers=4, num_decoder_layers=8
# Lighter encoding, more decoding iterations
# Use when: Dense object detection with heavy refinement

# Minimal (Demo)
num_encoder_layers=1, num_decoder_layers=1
# Fast testing, low accuracy
```

**Performance Comparison**:

| Config | Enc | Dec | Params | Speed | AP (COCO) | Use Case |
|--------|-----|-----|--------|-------|-----------|----------|
| Demo | 1 | 1 | ~26M | ~50 FPS | ~23% | Testing/Learning |
| Light | 2 | 2 | ~28M | ~45 FPS | ~30% | Fast inference |
| Medium | 4 | 4 | ~35M | ~35 FPS | ~36% | Balanced |
| Standard | 6 | 6 | ~41M | ~28 FPS | ~42% | Production (recommended) |
| Deep | 8 | 8 | ~47M | ~22 FPS | ~44% | High accuracy |

**AP = Average Precision** (detection accuracy, 0-100%, higher is better)

---

### 2. `nheads` (Attention Heads)

**Purpose**: Number of parallel attention mechanisms.

**Analogy**: Like having 8 people looking at the same image, each focusing on different aspects.

**How it works**:
```python
# With nheads=8 and hidden_dim=256:
# Each head gets 256/8 = 32 dimensions

Head 1: Focuses on object boundaries
Head 2: Focuses on textures
Head 3: Focuses on spatial relationships
Head 4: Focuses on colors
Head 5-8: Other patterns
```

**Constraint**:
```python
hidden_dim % nheads == 0  # Must be divisible!

# Valid combinations:
hidden_dim=256, nheads=8   ✓  (256/8 = 32 per head)
hidden_dim=256, nheads=4   ✓  (256/4 = 64 per head)
hidden_dim=512, nheads=16  ✓  (512/16 = 32 per head)

# Invalid combinations:
hidden_dim=256, nheads=7   ✗  (256/7 = 36.57... not integer)
hidden_dim=256, nheads=9   ✗  (256/9 = 28.44... not integer)
```

**Effect of Different Head Counts**:

| nheads | Dim per Head | Params | AP | Notes |
|--------|--------------|--------|-----|-------|
| 4 | 64 | ~38M | ~40% | Fewer heads, simpler attention |
| 8 | 32 | ~41M | ~42% | **Standard (recommended)** |
| 16 | 16 | ~52M | ~43% | More heads, richer features |

**Trade-off**:
- More heads → Richer feature representation
- More heads → More computation

**Relationship with Layers**:
- `nheads` applies to **every layer** (both encoder and decoder)
- All layers use the same number of heads

---

### 3. `num_queries`

**Purpose**: Number of "detection slots" - maximum objects the model can detect per image.

**Analogy**: Like having 100 detectors, each can specialize in finding different objects.

**Independence**: Completely independent from other parameters.

**How to Choose**:

```python
# Simple scenes (few objects)
num_queries=25
# Example: Detecting cars on highway (max ~10 cars per image)

# Standard scenes
num_queries=100
# COCO benchmark default
# Example: Street scenes with various objects

# Dense scenes (many objects)
num_queries=300
# Example: Crowd detection, dense retail shelves
```

**Effect on Performance**:

| num_queries | Memory | Speed | Max Objects | Use Case |
|-------------|--------|-------|-------------|----------|
| 25 | Low | Fast | 25 | Simple scenes |
| 100 | Medium | Medium | 100 | **Standard (COCO)** |
| 300 | High | Slow | 300 | Dense scenes |

**Important Notes**:
- If image has >num_queries objects, some will be missed
- If image has <num_queries objects, extra queries predict "background"
- Each query becomes a specialized detector during training

---

## Parameter Relationships

### Independence Matrix

| Parameter | Independent from | Constraint |
|-----------|------------------|------------|
| `num_encoder_layers` | All others | None |
| `num_decoder_layers` | All others | None |
| `nheads` | num_queries, layers | `hidden_dim % nheads == 0` |
| `num_queries` | All others | None |

**Key Insight**:
- ✅ Encoder and Decoder layers can be **different**
- ✅ nheads is **shared across all layers**
- ✅ num_queries is **completely independent**

---

## Configuration Examples

### Example 1: High Accuracy (Production)

```python
model = DETR(
    num_classes=91,
    num_encoder_layers=6,   # Deep feature extraction
    num_decoder_layers=6,   # Deep reasoning
    nheads=8,               # Standard
    num_queries=100,        # Standard COCO
    hidden_dim=256
)
# Expected: AP ~42%, ~28 FPS
```

### Example 2: Fast Inference

```python
model = DETR(
    num_classes=91,
    num_encoder_layers=4,   # Lighter encoding
    num_decoder_layers=3,   # Lighter decoding
    nheads=8,               # Keep standard
    num_queries=50,         # Fewer queries
    hidden_dim=256
)
# Expected: AP ~35%, ~40 FPS
```

### Example 3: Simple Scenes

```python
model = DETR(
    num_classes=10,         # Fewer classes
    num_encoder_layers=3,   # Shallow enough
    num_decoder_layers=3,
    nheads=8,
    num_queries=25,         # Fewer objects expected
    hidden_dim=256
)
# Expected: AP ~33%, ~45 FPS
```

### Example 4: Demo/Testing

```python
model = DETR(
    num_classes=3,
    num_encoder_layers=1,   # Minimal
    num_decoder_layers=1,   # Minimal
    nheads=8,
    num_queries=25,
    hidden_dim=256
)
# Expected: AP ~23%, ~50 FPS
# Use for: Quick prototyping, learning
```

---

## Usage

### Basic Usage

```python
from detr.model import DETR
import torch

# Create model
model = DETR(num_classes=91, num_queries=100)

# Forward pass
images = torch.randn(2, 3, 800, 800)  # [B, C, H, W]
output = model(images)

print(output['pred_logits'].shape)  # [2, 100, 92]
print(output['pred_boxes'].shape)   # [2, 100, 4]
```

### With Loss Function

```python
from detr.model import DETR
from detr.loss import HungarianMatcher, DETRLoss, compute_total_loss

# Model
model = DETR(num_classes=91, num_queries=100)

# Loss
weight_dict = {
    'class_weighting': 1.0,
    'bbox_weighting': 5.0,
    'giou_weighting': 2.0
}

matcher = HungarianMatcher(weight_dict)
criterion = DETRLoss(
    num_classes=91,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1
)

# Training loop
for images, targets in dataloader:
    predictions = model(images)
    loss_dict = criterion(predictions, targets)
    total_loss = compute_total_loss(loss_dict, weight_dict)

    total_loss.backward()
    optimizer.step()
```

### Load Pretrained Weights

```python
model = DETR(num_classes=91)
model.load_pretrained('detr_resnet50_coco.pth')
```

---

## Performance Tips

### 1. Layer Configuration

**For high accuracy**:
- Use 6+ encoder layers
- Use 6+ decoder layers
- Accept slower speed

**For fast inference**:
- Use 3-4 encoder layers
- Use 2-3 decoder layers
- Reduce num_queries

**For balanced**:
- Use 4-5 encoder layers
- Use 4-5 decoder layers
- Standard num_queries (100)

### 2. Attention Heads

**Recommendation**: Keep `nheads=8` unless you have specific needs.

**When to change**:
- `nheads=4`: Limited computation, simpler features
- `nheads=16`: More computation available, need richer features

### 3. Queries

**Rule of thumb**: `num_queries ≈ 2-3 × max_objects_per_image`

**Examples**:
- Person detection in meetings: 10-20 people → `num_queries=50`
- Street scenes: 20-40 objects → `num_queries=100`
- Crowd detection: 100+ people → `num_queries=300`

---

## Common Questions

### Q1: Must encoder and decoder have the same number of layers?

**No!** They are completely independent. You can use:
- 6 encoder + 3 decoder
- 4 encoder + 6 decoder
- Any combination

### Q2: What happens if I set nheads=7 with hidden_dim=256?

**Error!** You'll get a runtime error because 256/7 is not an integer. Always ensure `hidden_dim % nheads == 0`.

### Q3: Should I always use 100 queries?

**No.** Use based on your data:
- Simple scenes with few objects: 25-50 queries
- Standard scenes (COCO-like): 100 queries
- Dense scenes: 150-300 queries

### Q4: Does more layers always mean better accuracy?

**Yes, but with diminishing returns:**
- 1→3 layers: Large improvement (~10% AP)
- 3→6 layers: Medium improvement (~5% AP)
- 6→9 layers: Small improvement (~2% AP)

### Q5: How does nheads affect memory?

Marginally. Memory is dominated by:
1. Backbone (~60% of total)
2. Number of layers (~30%)
3. nheads (~10%)

---

## References

- **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Official Code**: https://github.com/facebookresearch/detr
- **Transformer Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
