"""
Evaluation script for sign detection model
"""

import argparse
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table

from sign_detection.models import SignDetectionModel
from sign_detection.data import SignDataset, get_val_transforms
from sign_detection.utils import calculate_metrics
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    """Evaluate model on dataset"""
    model.eval()

    all_outputs = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)

        all_outputs.append(outputs.cpu())
        all_targets.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_outputs, all_targets, num_classes)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate sign detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/images',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    console = Console()
    device = torch.device(args.device)

    # Load checkpoint
    console.print(f"[blue]Loading checkpoint from {args.checkpoint}[/blue]")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get('classes', None)

    # Load dataset
    transform = get_val_transforms()
    dataset = SignDataset(args.data_dir, classes=classes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    num_classes = len(dataset.classes)
    model = SignDetectionModel(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    console.print(f"[green]Model loaded successfully[/green]")
    console.print(f"[blue]Evaluating on {len(dataset)} samples[/blue]")

    # Evaluate
    metrics = evaluate(model, dataloader, device, num_classes)

    # Display results
    console.print(f"\n[bold green]Overall Accuracy: {metrics['accuracy']:.4f}[/bold green]\n")

    # Per-class accuracy table
    table = Table(title="Per-Class Accuracy")
    table.add_column("Class", style="cyan")
    table.add_column("Accuracy", style="green")

    for idx, acc in enumerate(metrics['class_accuracy']):
        class_name = dataset.get_class_name(idx)
        table.add_row(class_name, f"{acc:.4f}")

    console.print(table)


if __name__ == '__main__':
    main()
