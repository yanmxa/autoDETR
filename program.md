# autoDETR

Automated experimentation platform for DETR object detection. The LLM autonomously iterates on training configuration and architecture to improve detection accuracy.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autodetr/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autodetr/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `program.md` — this file, the experiment protocol.
   - `src/detr/config.py` — centralized configuration (model, training, evaluation).
   - `src/detr/train.py` — training loop. You modify this.
4. **Verify data exists**: Check that `data/train/` and `data/test/` contain images and labels.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

## Scope

**What you CAN modify:**
- `src/detr/config.py` — all hyperparameters: learning rate, batch size, loss weights, dropout, num_queries, encoder/decoder layers, scheduler, optimizer, eos_coef, etc.
- `src/detr/train.py` — training loop: augmentation strategy, gradient clipping, epoch count, warmup, etc.

**What you CANNOT modify:**
- `src/detr/model/detr_model.py` — the DETR architecture is fixed (ResNet50 + Transformer).
- `src/detr/loss/criterion.py` — loss computation (Hungarian matching + CE + L1 + GIoU).
- `src/detr/loss/matcher.py` — Hungarian matcher.
- `src/detr/data/dataset.py` — data loading pipeline.
- `src/detr/validate.py` — validation harness (the ground truth metric).
- Do not install new packages or add dependencies.

**The goal is simple: get the highest accuracy and lowest val_loss.** Everything within scope is fair game.

## The Experiment Loop

LOOP FOREVER:

1. Look at the current git state and `results.tsv` history.
2. Decide on an experimental change to `config.py` and/or `train.py`.
3. `git commit -s` the change with a descriptive message.
4. Run the experiment (training ends with metrics output automatically):
   ```bash
   autodetr-train > run.log 2>&1
   ```
5. Read out the results:
   ```bash
   grep "^val_loss:\|^accuracy:\|^mean_iou:\|^num_detections:" run.log
   ```
6. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in `results.tsv` (do NOT commit this file).
8. **If accuracy improved** (higher) or val_loss improved (lower):
   - Keep the commit, advance the branch.
   - Save a validation snapshot: `autodetr-val --tag <short_tag>`
   - This saves `val_results/val_<tag>_<number>.png` with auto-incrementing number per tag.
9. **If results are equal or worse**: `git reset --hard HEAD~1` to revert.

## Output Format

After validation, `autodetr-val` prints a summary:

Training automatically prints metrics at the end:

```
---
val_loss:         2.345678
accuracy:         0.7500
mean_iou:         0.6234
num_detections:   8
confidence_threshold: 0.10
---
```

When you run `autodetr-val --tag baseline` to save a snapshot, it also saves a visualization:
`val_results/val_baseline_01.png` (number auto-increments per tag: 01, 02, 03, ...)

## Logging Results

Log each experiment to `results.tsv` (tab-separated). Do NOT commit this file.

Header and columns:

```
commit	val_loss	accuracy	mean_iou	status	description
```

- `commit`: git short hash (7 chars)
- `val_loss`: validation loss (e.g. 2.345678), use 0.000000 for crashes
- `accuracy`: classification accuracy (0.0-1.0), use 0.000 for crashes
- `mean_iou`: mean IoU of matched boxes, use 0.000 for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what this experiment tried

Example:

```
commit	val_loss	accuracy	mean_iou	status	description
a1b2c3d	2.345	0.750	0.623	keep	baseline
b2c3d4e	2.100	0.810	0.670	keep	increase LR to 1e-3
c3d4e5f	2.500	0.700	0.580	discard	switch to AdamW
d4e5f6g	0.000	0.000	0.000	crash	double model width (OOM)
```

To visualize experiment history: `autodetr-plot`

## Simplicity Criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

- 0.01 accuracy improvement that adds 20 lines of hacky code? Probably not worth it.
- 0.01 accuracy improvement from deleting code? Definitely keep.
- Same accuracy but simpler code? Keep.

## Current Configuration (Baseline)

| Parameter | Value |
|-----------|-------|
| Classes | 3 (one, two, three) |
| Encoder layers | 1 |
| Decoder layers | 1 |
| Object queries | 25 |
| Hidden dim | 256 |
| Attention heads | 8 |
| Dropout | 0.2 |
| Epochs | 150 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Loss weights | class=2.0, bbox=10.0, giou=5.0 |
| Training samples | 170 |
| Test samples | 32 |

## Commands

| Command | Description |
|---------|-------------|
| `autodetr-train` | Train DETR model |
| `autodetr-val` | Run validation and generate visualization |
| `autodetr-eval` | Interactive evaluation with visualization |
| `autodetr-plot` | Plot experiment history from results.tsv |
| `autodetr-collect` | Capture training images from webcam |

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep or away. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical hyperparameter changes, try different scheduler strategies. The loop runs until the human interrupts you.
