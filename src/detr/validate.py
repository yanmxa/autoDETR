"""
autoDETR Validation Script

Runs validation on a fixed set of test images, computes metrics (val_loss, accuracy,
mean_iou), and optionally generates a 2x5 grid visualization of predictions.

Usage:
    autodetr-val                        # metrics only (no image saved)
    autodetr-val --tag baseline         # metrics + save val_results/val_baseline_01.png
"""

import argparse
import subprocess
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from detr.data import DETRDataset, collate_fn
from detr.model import DETR
from detr.loss import DETRLoss, HungarianMatcher, compute_total_loss
from detr.config import get_model_config, get_evaluation_config
from detr.evaluate import rescale_bboxes, filter_predictions


CLASS_NAMES = ['one', 'two', 'three']
NUM_VAL_IMAGES = 10


def get_next_experiment_number(tag, save_dir='val_results'):
    """Get the next sequential number for a given tag by scanning existing files."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return 1
    # Match files like val_<tag>_01.png, val_<tag>_02.png
    existing = list(save_dir.glob(f'val_{tag}_*.png'))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        # Extract number from end: val_baseline_03.png -> 03
        last_part = f.stem.rsplit('_', 1)[-1]
        try:
            numbers.append(int(last_part))
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def compute_accuracy_and_iou(predictions, targets, matcher):
    """
    Compute classification accuracy and mean IoU using Hungarian matching.

    Returns:
        accuracy: fraction of correctly classified matched objects
        mean_iou: average IoU of matched prediction-target box pairs
        total_matched: total number of matched objects
    """
    indices = matcher(predictions, targets)

    correct = 0
    total = 0
    ious = []

    pred_logits = predictions['pred_logits']
    pred_boxes = predictions['pred_boxes']

    for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue

        # Classification accuracy
        pred_classes = pred_logits[batch_idx, pred_idx].argmax(-1)
        gt_classes = targets[batch_idx]['labels'][tgt_idx]
        correct += (pred_classes == gt_classes).sum().item()
        total += len(pred_idx)

        # IoU for matched boxes
        p_boxes = pred_boxes[batch_idx, pred_idx]
        t_boxes = targets[batch_idx]['boxes'][tgt_idx]

        # Convert cxcywh to xyxy
        p_xyxy = _cxcywh_to_xyxy(p_boxes)
        t_xyxy = _cxcywh_to_xyxy(t_boxes)

        for p, t in zip(p_xyxy, t_xyxy):
            iou = _compute_iou(p, t)
            ious.append(iou)

    accuracy = correct / total if total > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return accuracy, mean_iou, total


def _cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def _compute_iou(box1, box2):
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def generate_val_visualization(images, predictions, targets, config, tag, save_dir='val_results'):
    """
    Generate 2x5 grid visualization of predictions on fixed validation images.

    Args:
        images: tensor [N, 3, H, W]
        predictions: dict with pred_logits and pred_boxes
        targets: list of target dicts
        config: evaluation config
        tag: experiment tag for filename
        save_dir: directory to save visualization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = min(NUM_VAL_IMAGES, len(images))
    rows, cols = 2, 5

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    # Filter predictions
    threshold = config.get('confidence_threshold', 0.10)
    batch_indices, _, classes, probas, bboxes = filter_predictions(
        predictions['pred_logits'][:n],
        predictions['pred_boxes'][:n],
        confidence_threshold=threshold,
        top_k=1
    )

    # Rescale boxes
    img_size = (config.get('image_size', 224), config.get('image_size', 224))
    bboxes_scaled = rescale_bboxes(bboxes, img_size)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(n):
        ax = axes[idx]

        # Denormalize and display image
        img_np = images[idx].permute(1, 2, 0).cpu().numpy()
        img_np = img_np * std + mean
        img_np = img_np.clip(0, 1)
        ax.imshow(img_np)

        # Draw ground truth boxes (blue, dashed)
        gt_boxes = targets[idx]['boxes'].cpu().numpy()
        gt_labels = targets[idx]['labels'].cpu().numpy()
        h, w = img_size
        for box, label in zip(gt_boxes, gt_labels):
            cx, cy, bw, bh = box
            x1, y1 = (cx - bw/2) * w, (cy - bh/2) * h
            from matplotlib.patches import Rectangle
            rect = Rectangle((x1, y1), bw * w, bh * h,
                            fill=False, color='#4488FF', linewidth=1.5, linestyle='--')
            ax.add_patch(rect)

        # Draw predictions (green, solid)
        for bi, cls, prob, bbox in zip(batch_indices, classes, probas, bboxes_scaled):
            if bi.item() != idx:
                continue
            xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()
            from matplotlib.patches import Rectangle
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, color='#00FF00', linewidth=2)
            ax.add_patch(rect)
            label = f'{CLASS_NAMES[cls.item()]}: {prob:.2f}'
            ax.text(xmin, ymin - 3, label, fontsize=8, color='white',
                    bbox=dict(facecolor='green', alpha=0.7, boxstyle='round,pad=0.2'))

        # Draw GT label in corner
        gt_class_str = ', '.join(CLASS_NAMES[l] for l in gt_labels)
        ax.set_title(f'GT: {gt_class_str}', fontsize=9)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx].axis('off')

    num = get_next_experiment_number(tag, str(save_dir))
    filename = f'val_{tag}_{num:02d}.png'

    fig.suptitle(f'{tag} #{num:02d}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(save_path)


def run_validation(tag=None, quiet=False):
    """
    Run full validation pipeline.

    Args:
        tag: experiment tag. If provided, saves visualization to val_results/val_<num>_<tag>.png.
             If None, only computes and prints metrics (no image saved).
        quiet: if True, suppress print output

    Returns:
        dict with val_loss, accuracy, mean_iou, num_detections, save_path
    """

    config = get_evaluation_config()
    model_config = get_model_config()
    device = torch.device(config['device'])

    # Load test dataset
    test_dataset = DETRDataset(config['test_data_dir'], train=False)

    # Fixed validation subset: first NUM_VAL_IMAGES by sorted label filename
    sorted_indices = sorted(range(len(test_dataset)),
                           key=lambda i: test_dataset.labels[i])
    val_indices = sorted_indices[:NUM_VAL_IMAGES]
    val_subset = Subset(test_dataset, val_indices)

    val_loader = DataLoader(val_subset, batch_size=NUM_VAL_IMAGES,
                           collate_fn=collate_fn, shuffle=False)
    full_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            collate_fn=collate_fn, shuffle=False)

    # Create model and load checkpoint
    model = DETR(
        num_classes=model_config['num_classes'],
        hidden_dim=model_config['hidden_dim'],
        nheads=model_config['nheads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_queries=model_config['num_queries'],
        dropout=model_config['dropout'],
        verbose=False
    )

    checkpoint_path = Path(config['checkpoint_path'])
    if checkpoint_path.exists():
        state_dict = torch.load(str(checkpoint_path), map_location='cpu')
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # Setup loss criterion for val_loss
    from detr.config import get_training_config
    train_config = get_training_config()
    matcher = HungarianMatcher(train_config['loss_weights'])
    criterion = DETRLoss(
        num_classes=model_config['num_classes'],
        matcher=matcher,
        weight_dict=train_config['loss_weights'],
        eos_coef=train_config['eos_coef']
    )

    # 1. Compute val_loss on full test set
    total_loss = 0.0
    total_batches = 0
    all_accuracy_correct = 0
    all_accuracy_total = 0
    all_ious = []

    with torch.no_grad():
        for images, targets in full_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss = compute_total_loss(loss_dict, criterion.weight_dict)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                total_batches += 1

            # Accuracy and IoU via Hungarian matching
            acc, miou, matched = compute_accuracy_and_iou(predictions, targets, matcher)
            all_accuracy_correct += int(acc * matched)
            all_accuracy_total += matched
            if matched > 0:
                all_ious.append(miou)

    val_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    accuracy = all_accuracy_correct / all_accuracy_total if all_accuracy_total > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    # 2. Count detections
    threshold = config.get('confidence_threshold', 0.10)
    num_detections = 0
    with torch.no_grad():
        for images, targets in full_loader:
            images = images.to(device)
            preds = model(images)
            probs = preds['pred_logits'].softmax(-1)[:, :, :-1]
            max_probs, _ = probs.max(-1)
            num_detections += (max_probs > threshold).sum().item()

    # 3. Generate visualization only when tag is provided (i.e., a "keep" experiment)
    save_path = None
    if tag is not None:
        with torch.no_grad():
            val_images, val_targets = next(iter(val_loader))
            val_images = val_images.to(device)
            val_preds = model(val_images)

        save_path = generate_val_visualization(
            val_images.cpu(),
            {k: v.cpu() for k, v in val_preds.items()},
            val_targets, config, tag
        )

    # 4. Print structured output
    results = {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'num_detections': num_detections,
        'confidence_threshold': threshold,
        'experiment_tag': tag or 'n/a',
        'save_path': save_path,
    }

    if not quiet:
        print("---")
        print(f"val_loss:         {val_loss:.6f}")
        print(f"accuracy:         {accuracy:.4f}")
        print(f"mean_iou:         {mean_iou:.4f}")
        print(f"num_detections:   {num_detections}")
        print(f"confidence_threshold: {threshold}")
        if save_path:
            print(f"val_image:        {save_path}")
        print("---")

    return results


def main():
    parser = argparse.ArgumentParser(description='autoDETR Validation')
    parser.add_argument('--tag', type=str, default=None,
                       help='Experiment tag (default: git branch name)')
    args = parser.parse_args()

    run_validation(tag=args.tag)


if __name__ == '__main__':
    main()
