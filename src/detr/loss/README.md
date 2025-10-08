# DETR Loss Function

Implementation of DETR (Detection Transformer) loss function, including Hungarian matcher and complete loss computation.

## File Structure

```
loss/
├── __init__.py          # Module exports
├── matcher.py           # Hungarian matcher
├── criterion.py         # DETR loss criterion
├── example.py           # Usage example
└── README.md            # This file
```

## Quick Start

```bash
cd src/detr/loss
python example.py
```

## Core Components

### 1. Hungarian Matcher

Computes optimal bipartite matching between predictions and ground truth using the Hungarian algorithm.

**Cost function:**
```python
cost = w_class * cost_class + w_bbox * cost_bbox + w_giou * cost_giou
```

where:
- `cost_class`: Negative probability of correct class prediction
- `cost_bbox`: L1 distance between box coordinates
- `cost_giou`: Negative Generalized IoU

### 2. DETR Loss

Three loss components:
- **Classification Loss**: Weighted cross-entropy (lower weight for "no-object" class)
- **L1 Box Loss**: Mean absolute error between box coordinates
- **GIoU Loss**: Generalized IoU loss for scale-invariant box regression

## Usage

```python
from detr.loss import HungarianMatcher, DETRLoss, compute_total_loss

# Configuration
num_classes = 91
weight_dict = {
    'class_weighting': 1.0,
    'bbox_weighting': 5.0,
    'giou_weighting': 2.0
}

# Initialize
matcher = HungarianMatcher(weight_dict)
criterion = DETRLoss(
    num_classes=num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1
)

# Compute loss
loss_dict = criterion(predictions, targets)
total_loss = compute_total_loss(loss_dict, weight_dict)
```

## Data Format

### Predictions

```python
predictions = {
    'pred_logits': torch.Tensor,  # [B, num_queries, num_classes+1]
    'pred_boxes': torch.Tensor    # [B, num_queries, 4] in cxcywh format
}
```

### Targets

```python
targets = [
    {
        'labels': torch.Tensor,  # [num_objects] - class indices
        'boxes': torch.Tensor    # [num_objects, 4] in cxcywh format
    },
    # ... one dict per image
]
```

**Box format**: `(center_x, center_y, width, height)` normalized to [0, 1]

## Key Concepts

### Hungarian Matching

DETR uses a fixed set of object queries (e.g., 100) to predict all objects. The Hungarian algorithm finds the optimal one-to-one matching between predictions and ground truth based on the cost matrix.

**Example:**
```
Predictions: 100 queries
Ground truth: 3 objects

Cost matrix C[i,j] = cost of matching query i to object j
Hungarian algorithm → Optimal assignment: [(q5, gt0), (q23, gt1), (q67, gt2)]
```

### Generalized IoU (GIoU)

GIoU extends standard IoU to provide meaningful gradients even when boxes don't overlap:

```
GIoU = IoU - (C \ (A ∪ B)) / C

where:
- A, B: two bounding boxes
- C: smallest box enclosing both A and B
```

**Properties:**
- Range: [-1, 1]
- GIoU = 1: perfect overlap
- GIoU < 0: non-overlapping boxes
- Provides gradients when IoU = 0

### Loss Normalization

All losses are normalized by the total number of objects in the batch:

```python
num_boxes = sum(len(t['labels']) for t in targets)
loss = loss.sum() / num_boxes
```

This ensures loss magnitude is independent of batch size.

## Hyperparameters

### Loss Weights

```python
weight_dict = {
    'class_weighting': 1.0,   # Classification loss weight
    'bbox_weighting': 5.0,    # L1 box loss weight
    'giou_weighting': 2.0     # GIoU loss weight
}
```

**Tuning tips:**
- Increase `bbox_weighting` and `giou_weighting` for better localization
- Increase `giou_weighting` for better small object detection

### EOS Coefficient

```python
eos_coef = 0.1  # Weight for "no-object" class
```

**Effect:**
- Lower value (0.05): fewer false positives
- Higher value (0.2): fewer false negatives

## Implementation Details

### Box Coordinate Conversion

**cxcywh → xyxy:**
```python
x1 = cx - w/2
y1 = cy - h/2
x2 = cx + w/2
y2 = cy + h/2
```

### Class Imbalance Handling

Most queries predict "no-object" (e.g., 97 out of 100). To handle this:
- Use lower weight for "no-object" class (`eos_coef=0.1`)
- Similar effect to focal loss

### Matched vs Unmatched Queries

- **Matched queries**: Supervised with object class + box regression
- **Unmatched queries**: Supervised with "no-object" class only (no box loss)

## References

- **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **GIoU Paper**: [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)
- **Hungarian Algorithm**: [Wikipedia](https://en.wikipedia.org/wiki/Hungarian_algorithm)
