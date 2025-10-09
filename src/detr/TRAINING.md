# DETR Training Configuration Guide

Training configuration and workflow explanation for DETR object detection.

---

## Quick Start

```bash
cd src/detr
python train.py
```

Edit `get_training_config()` in `train.py` to modify parameters.

---

## Training Configuration

### Learning Rate Scheduler

```python
'optimizer': 'Adam',                    # Optimizer type: 'Adam' or 'AdamW'
'scheduler': 'CosineAnnealingWarmRestarts',
'T_0': None,                            # First restart period (steps)
'T_mult': 2,                            # Period multiplier
```

#### What is `T_0`?

**T_0** = First cycle length in **steps** (not epochs)

```python
T_0 = len(train_dataloader) * 30  # Auto-computed if None
# Example: 120 batches/epoch × 30 epochs = 3600 steps
```

**Purpose**: Learning rate decreases from initial value to minimum over `T_0` steps, then "restarts" (jumps back to initial value).

#### What is `T_mult`?

**T_mult** = Period multiplier after each restart

```python
T_mult = 2

# Cycle 1: T_0 steps
# Cycle 2: T_0 × 2 steps
# Cycle 3: T_0 × 4 steps
# Cycle 4: T_0 × 8 steps
# ... cycles get progressively longer
```

**Purpose**: Later cycles are longer, giving model more time to refine.

#### Visualization

```
Learning Rate (T_mult=2, T_0=3600 steps):

LR ↑
   |     ╱╲         ╱──╲              ╱────────╲
   | max ╱  ╲       ╱    ╲            ╱          ╲
   |    ╱    ╲     ╱      ╲          ╱            ╲
   | min╱      ╲___╱        ╲________╱              ╲
   |_________________________________________________→ Steps
     ←─T_0─→   ←──2×T_0──→  ←─────4×T_0──────→
     (30ep)      (60ep)         (120ep)

Restarts: ↑       ↑              ↑
```

**Why restarts?**
- Escape local minima by periodically increasing LR
- Multiple chances to find better solutions
- Early cycles: exploration
- Late cycles: refinement

#### Parameter Tuning

| Scenario | T_0 | T_mult | Effect |
|----------|-----|--------|--------|
| Fast experiments | `len(dl) * 10` | 2 | Frequent restarts, quick exploration |
| **Standard (recommended)** | **`len(dl) * 30`** | **2** | **Balanced exploration/refinement** |
| Long training | `len(dl) * 50` | 2 | Fewer restarts, stable convergence |
| Constant cycles | `len(dl) * 20` | 1 | No growth, equal exploration |
| Aggressive growth | `len(dl) * 20` | 3 | Quick exploration → long refinement |

---

### Checkpoint Configuration

```python
'pretrained_path': None,           # Path to pretrained weights
'checkpoint_dir': 'checkpoints',   # Directory to save checkpoints
'save_interval': 10,               # Save every N epochs
```

#### Pretrained Weights

```python
# Load pretrained model (fine-tuning)
'pretrained_path': 'pretrained/detr_coco.pt'

# Train from scratch
'pretrained_path': None
```

**Behavior**:
- `None`: Random initialization, displays info message
- Path string: Attempts to load weights
  - Success: Uses pretrained weights
  - Failure: Displays error, continues with random init (doesn't crash)

#### Checkpoint Saving

Automatically saves three types of checkpoints:

1. **Best model** (lowest test loss)
   ```
   checkpoints/epoch_best.pt
   ```

2. **Periodic checkpoints** (every `save_interval` epochs)
   ```
   checkpoints/epoch_10.pt
   checkpoints/epoch_20.pt
   checkpoints/epoch_30.pt
   ```

3. **Final model** (end of training)
   ```
   checkpoints/epoch_final.pt
   ```

**Disable saving**:
```python
'checkpoint_dir': None  # No checkpoints saved
```

---

## Training Workflow

### 1. Initialization Phase

```
┌─────────────────────────────────────┐
│ Load Configuration                  │
│ • Read get_training_config()        │
│ • Display config tables             │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Create Datasets                     │
│ • Train: DETRDataset(train=True)    │
│ • Test: DETRDataset(train=False)    │
│ • Show dataset statistics           │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Create DataLoaders                  │
│ • Batch size: 4                     │
│ • Custom collate_fn for targets     │
│ • Display batch counts              │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Initialize Model                    │
│ • Create DETR model                 │
│ • Load pretrained weights (if set)  │
│ • Move to device (GPU/CPU)          │
│ • Display model info                │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Setup Training Components           │
│ • Optimizer: Adam/AdamW             │
│ • Scheduler: CosineAnnealingWR      │
│ • Matcher: HungarianMatcher         │
│ • Criterion: DETRLoss               │
└─────────────────────────────────────┘
```

### 2. Training Loop (Per Epoch)

```
For each epoch (1 to 100):
    ┌─────────────────────────────────────────┐
    │ TRAINING PHASE                          │
    │ ─────────────────────────────────────── │
    │ model.train()                           │
    │                                         │
    │ For each batch in train_dataloader:    │
    │   1. Load images & targets              │
    │   2. Move to device (GPU/CPU)           │
    │   3. Forward pass: predictions = model(images)
    │   4. Compute loss:                      │
    │      • Hungarian matching               │
    │      • Classification loss (CE)         │
    │      • BBox regression loss (L1)        │
    │      • GIoU loss                        │
    │   5. Backward pass: loss.backward()     │
    │   6. Optimizer step                     │
    │   7. Update progress bar                │
    │                                         │
    │ Result: avg_train_loss                  │
    └─────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │ EVALUATION PHASE                        │
    │ ─────────────────────────────────────── │
    │ model.eval()                            │
    │ with torch.no_grad():                   │
    │                                         │
    │   For each batch in test_dataloader:   │
    │     1. Load images & targets            │
    │     2. Move to device                   │
    │     3. Forward pass (no gradients)      │
    │     4. Compute loss                     │
    │     5. Accumulate loss                  │
    │                                         │
    │ Result: avg_test_loss                   │
    └─────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │ POST-EPOCH OPERATIONS                   │
    │ ─────────────────────────────────────── │
    │ 1. scheduler.step()                     │
    │    • Update learning rate               │
    │    • Handle warm restarts               │
    │                                         │
    │ 2. Check if best model:                 │
    │    if test_loss < best_test_loss:       │
    │      Save checkpoints/epoch_best.pt     │
    │                                         │
    │ 3. Periodic checkpoint:                 │
    │    if (epoch+1) % save_interval == 0:   │
    │      Save checkpoints/epoch_N.pt        │
    │                                         │
    │ 4. Update progress display              │
    └─────────────────────────────────────────┘
           ↓
    (Repeat for next epoch)
```

### 3. Finalization Phase

```
┌─────────────────────────────────────┐
│ Save Final Model                    │
│ • checkpoints/epoch_final.pt        │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Display Training Summary            │
│ • Total epochs completed            │
│ • Best test loss achieved           │
│ • Training complete message         │
└─────────────────────────────────────┘
```

---

## Progress Display

During training, you'll see:

```
🏗️  Model Configuration
┌────────────────────┬────────┐
│ Parameter          │ Value  │
├────────────────────┼────────┤
│ Number of Classes  │ 3      │
│ Object Queries     │ 100    │
│ Encoder Layers     │ 1      │
│ Decoder Layers     │ 1      │
│ Attention Heads    │ 8      │
└────────────────────┴────────┘

🏋️  Training Configuration
┌──────────────────┬──────────┐
│ Parameter        │    Value │
├──────────────────┼──────────┤
│ Total Epochs     │      100 │
│ Batch Size       │        4 │
│ Learning Rate    │    1e-05 │
│ Optimizer        │     Adam │
│ Scheduler        │ Cosine.. │
│ Device           │     CUDA │
└──────────────────┴──────────┘

⚖️  Loss Weights
┌──────────────────────┬────────┐
│ Loss Component       │ Weight │
├──────────────────────┼────────┤
│ Classification       │    1.0 │
│ BBox Regression (L1) │    5.0 │
│ GIoU                 │    2.0 │
│ EOS Coefficient      │   0.10 │
└──────────────────────┴────────┘

─────────────────────────────────────────
Starting Training
─────────────────────────────────────────

⠋ Epoch 25/100 ━━━━━━━━━━━━ 45/120 • Train Loss: 2.34567 • Test Loss: 2.45678 • 0:01:23

✓ Checkpoint saved: checkpoints/epoch_best.pt
✓ Checkpoint saved: checkpoints/epoch_50.pt
```

---

## Error Handling

If training fails, you'll see:

```
❌ Error
─────────────────────────────────────
Training Error at Epoch 25

RuntimeError: CUDA out of memory
─────────────────────────────────────

[Full stack trace displayed]
```

**Common fixes**:
- Reduce batch size
- Reduce model size (fewer layers, smaller hidden_dim)
- Use smaller image size

---

## Example Configurations

### Quick Test (1-2 hours)

```python
config = {
    'num_encoder_layers': 1,
    'num_decoder_layers': 1,
    'num_queries': 50,
    'epochs': 50,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'T_0': len(train_dataloader) * 10,
    'T_mult': 2,
    'pretrained_path': None
}
```

### Standard Training (1 day)

```python
config = {
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'num_queries': 100,
    'epochs': 300,
    'batch_size': 4,
    'learning_rate': 1e-5,
    'T_0': len(train_dataloader) * 30,  # Auto-computed
    'T_mult': 2,
    'pretrained_path': 'pretrained/detr_resnet50.pt'
}
```

---

## References

- **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **SGDR Paper**: [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
