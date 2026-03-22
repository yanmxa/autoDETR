"""
DETR Evaluation Script

This script evaluates the trained DETR model on test data and visualizes predictions.
"""

import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from detr.data import DETRDataset, collate_fn
from detr.model import DETR
from detr.config import get_evaluation_config
from detr.utils import print_message, print_empty_line


def rescale_bboxes(boxes, img_size):
    """
    Rescale bounding boxes from normalized [0, 1] to pixel coordinates.

    Args:
        boxes: Tensor of shape [N, 4] in format [cx, cy, w, h] normalized
        img_size: Tuple (height, width) of target image size

    Returns:
        Tensor of shape [N, 4] in format [xmin, ymin, xmax, ymax] in pixels
    """
    h, w = img_size

    # Convert from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # xmin = cx - w/2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # ymin = cy - h/2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # xmax = cx + w/2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # ymax = cy + h/2

    # Scale to image size
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], dtype=boxes.dtype)

    return boxes_xyxy


def filter_predictions(pred_logits, pred_boxes, confidence_threshold=0.7, top_k=None):
    """
    Filter predictions based on confidence threshold and optionally select top-k.

    Args:
        pred_logits: Predicted class logits [batch, num_queries, num_classes+1]
        pred_boxes: Predicted boxes [batch, num_queries, 4]
        confidence_threshold: Minimum confidence score
        top_k: If specified, keep only top-k highest confidence detections per image
               If 1, only the highest confidence detection per image is kept
               If None, all detections above threshold are kept

    Returns:
        Tuple of (batch_indices, query_indices, classes, probas, boxes)
    """
    # Get class probabilities (excluding background class)
    probabilities = pred_logits.softmax(-1)[:, :, :-1]

    # Get max probability and class for each query
    max_probs, max_classes = probabilities.max(-1)

    # Filter by confidence threshold
    keep_mask = max_probs > confidence_threshold
    batch_indices, query_indices = torch.where(keep_mask)

    # Extract filtered results
    bboxes = pred_boxes[batch_indices, query_indices, :]
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        batch_size = pred_logits.shape[0]

        # Process each image separately
        new_batch_indices = []
        new_query_indices = []
        new_classes = []
        new_probas = []
        new_bboxes = []

        for batch_idx in range(batch_size):
            # Get detections for this image
            img_mask = batch_indices == batch_idx
            img_probas = probas[img_mask]

            if len(img_probas) == 0:
                continue

            # Get top-k highest confidence detections
            k = min(top_k, len(img_probas))
            top_k_indices = torch.topk(img_probas, k).indices

            # Collect top-k detections
            img_batch_indices = batch_indices[img_mask][top_k_indices]
            img_query_indices = query_indices[img_mask][top_k_indices]
            img_classes = classes[img_mask][top_k_indices]
            img_probas_topk = img_probas[top_k_indices]
            img_bboxes = bboxes[img_mask][top_k_indices]

            new_batch_indices.append(img_batch_indices)
            new_query_indices.append(img_query_indices)
            new_classes.append(img_classes)
            new_probas.append(img_probas_topk)
            new_bboxes.append(img_bboxes)

        # Concatenate results
        if new_batch_indices:
            batch_indices = torch.cat(new_batch_indices)
            query_indices = torch.cat(new_query_indices)
            classes = torch.cat(new_classes)
            probas = torch.cat(new_probas)
            bboxes = torch.cat(new_bboxes)
        else:
            # No detections
            batch_indices = torch.tensor([], dtype=torch.long)
            query_indices = torch.tensor([], dtype=torch.long)
            classes = torch.tensor([], dtype=torch.long)
            probas = torch.tensor([], dtype=torch.float32)
            bboxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)

    return batch_indices, query_indices, classes, probas, bboxes


def visualize_predictions(images, batch_indices, classes, probas, bboxes,
                         class_names, img_size, save_path=None, top_k=None):
    """
    Visualize model predictions on images.

    Args:
        images: Batch of images [batch, 3, H, W]
        batch_indices: Which image each detection belongs to
        classes: Predicted class indices
        probas: Confidence scores
        bboxes: Bounding boxes in pixel coordinates
        class_names: List of class names
        img_size: Original image size (H, W)
        save_path: Optional path to save visualization
        top_k: If specified, indicates top-k filtering was applied
    """
    import math

    num_images = len(images)

    # Calculate optimal grid layout
    # Try to make grid as square as possible
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Create figure with dynamic size
    fig_width = min(cols * 4, 20)  # Max 20 inches wide
    fig_height = min(rows * 4, 20)  # Max 20 inches tall

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle single row/column case
    if num_images == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (img, ax) in enumerate(zip(images, axes)):
        if idx >= num_images:
            break

        # Convert from [C, H, W] to [H, W, C] and denormalize
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # Denormalize (ImageNet stats)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_np = img_np * std + mean
        img_np = img_np.clip(0, 1)

        ax.imshow(img_np)
        ax.axis('off')

        # Draw predictions for this image
        num_detections = 0
        for batch_idx, box_class, box_prob, bbox in zip(
            batch_indices, classes, probas, bboxes
        ):
            if batch_idx == idx:
                xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()

                # Draw bounding box
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    fill=False, color='#00FF00', linewidth=2
                )
                ax.add_patch(rect)

                # Add label
                label = f'{class_names[box_class]}: {box_prob:.2f}'
                ax.text(
                    xmin, ymin - 5, label,
                    fontsize=10,
                    bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round,pad=0.3'),
                    verticalalignment='bottom'
                )
                num_detections += 1

        # Create title based on filtering mode
        if top_k == 1:
            title = f'Image {idx + 1} - Highest confidence only'
        elif top_k is not None:
            title = f'Image {idx + 1} - Top-{top_k} ({num_detections} det.)'
        else:
            title = f'Image {idx + 1} - {num_detections} detections'
        ax.set_title(title, fontsize=12)

    # Hide unused subplots
    total_subplots = rows * cols
    for idx in range(num_images, total_subplots):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_message(f"[green]✓[/green] Visualization saved to: [cyan]{save_path}[/cyan]")

    plt.show()


def evaluate_model(config):
    """
    Main evaluation function.

    Args:
        config: Evaluation configuration dictionary
    """
    print_message("\n[bold cyan]DETR Model Evaluation[/bold cyan]\n")

    # Load test dataset
    print_message("[bold]Loading test dataset...[/bold]")
    test_dataset = DETRDataset(config['test_data_dir'], train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    print_message(f"[green]✓[/green] Test dataset loaded: [yellow]{len(test_dataset)}[/yellow] images")
    print_message(f"[green]✓[/green] Test batches: [yellow]{len(test_dataloader)}[/yellow]")
    print_empty_line()

    # Create model
    print_message("[bold]Initializing model...[/bold]")
    model = DETR(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        nheads=config['nheads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_queries=config['num_queries'],
        dropout=config['dropout'],
        verbose=False  # Don't print model info during eval
    )

    # Load checkpoint
    checkpoint_path = Path(config['checkpoint_path'])
    if not checkpoint_path.exists():
        print_message(f"[red]✗[/red] Checkpoint not found: {checkpoint_path}")
        print_message("[yellow]⚠[/yellow] Using randomly initialized model (for testing only)")
    else:
        print_message(f"[bold]Loading checkpoint from:[/bold] [cyan]{checkpoint_path}[/cyan]")
        model.load_pretrained(str(checkpoint_path))

    # Move to device and set to eval mode
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    print_message(f"[green]✓[/green] Model ready on device: [yellow]{device}[/yellow]")
    print_empty_line()

    # Collect images to evaluate
    num_images = config.get('num_images', 4)
    batches_needed = (num_images + config['batch_size'] - 1) // config['batch_size']

    print_message(f"[bold]Collecting {num_images} images ({batches_needed} batch(es))...[/bold]")

    all_images = []
    all_predictions = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_dataloader):
            if batch_idx >= batches_needed:
                break

            images = images.to(device)
            predictions = model(images)

            all_images.append(images)
            all_predictions.append(predictions)

    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Concatenate all batches
    images = torch.cat(all_images, dim=0)[:num_images]  # Trim to exact number
    pred_logits = torch.cat([p['pred_logits'] for p in all_predictions], dim=0)[:num_images]
    pred_boxes = torch.cat([p['pred_boxes'] for p in all_predictions], dim=0)[:num_images]

    predictions = {
        'pred_logits': pred_logits,
        'pred_boxes': pred_boxes
    }

    print_message(f"[green]✓[/green] Inference completed on {len(images)} images in [yellow]{inference_time:.2f}ms[/yellow]")
    print_message(f"[blue]ℹ[/blue] Avg time per image: [yellow]{inference_time/len(images):.2f}ms[/yellow]")
    print_empty_line()

    # Filter predictions
    top_k = config.get('top_k', None)
    if top_k is not None:
        print_message(f"[bold]Filtering predictions (threshold: {config['confidence_threshold']}, top-{top_k} per image)...[/bold]")
    else:
        print_message(f"[bold]Filtering predictions (threshold: {config['confidence_threshold']})...[/bold]")

    batch_indices, query_indices, classes, probas, bboxes = filter_predictions(
        predictions['pred_logits'],
        predictions['pred_boxes'],
        confidence_threshold=config['confidence_threshold'],
        top_k=top_k
    )

    # Rescale boxes to image coordinates
    img_size = (config.get('image_size', 224), config.get('image_size', 224))
    bboxes_scaled = rescale_bboxes(bboxes, img_size)

    print_message(f"[green]✓[/green] Found [yellow]{len(classes)}[/yellow] detections across [yellow]{len(images)}[/yellow] images")
    print_empty_line()

    # Print detection details
    print_message("[bold]Detection Results:[/bold]")
    class_names = ['one', 'two', 'three']  # TODO: Load from config

    for i in range(len(classes)):
        batch_idx = batch_indices[i].item()
        class_name = class_names[classes[i].item()]
        confidence = probas[i].item()
        bbox = bboxes_scaled[i].detach().cpu().numpy()

        print_message(
            f"  [dim]Image {batch_idx}:[/dim] "
            f"[cyan]{class_name}[/cyan] "
            f"([yellow]{confidence:.3f}[/yellow]) "
            f"[dim]bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}][/dim]"
        )
    print_empty_line()

    # Visualize results
    print_message("[bold]Generating visualization...[/bold]")
    visualize_predictions(
        images.cpu(),
        batch_indices.cpu(),
        classes.cpu(),
        probas.cpu(),
        bboxes_scaled.cpu(),
        class_names,
        img_size,
        save_path=config['save_path'],
        top_k=top_k
    )

    print_message("\n[bold green]✓ Evaluation complete![/bold green]\n")


def main():
    """Main entry point."""
    config = get_evaluation_config()
    evaluate_model(config)


if __name__ == '__main__':
    main()
