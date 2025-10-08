"""
Training script for sign detection model
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from sign_detection.models import SignDetectionModel
from sign_detection.data import SignDataset, get_train_transforms, get_val_transforms
from sign_detection.utils import setup_logger, calculate_accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    console: Console
) -> float:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"Epoch {epoch}", total=len(dataloader))

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            acc = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_acc += acc

            # Update progress
            progress.update(task, advance=1)

            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/Acc_Step', acc, global_step)

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        running_loss += loss.item()
        running_acc += acc

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train sign detection model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/images',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    # Setup
    console = Console()
    logger = setup_logger('train')
    device = torch.device(args.device)

    console.print(f"[bold green]Training on device: {device}[/bold green]")

    # Load dataset
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    dataset = SignDataset(args.data_dir, transform=train_transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    console.print(f"[blue]Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}[/blue]")

    # Create model
    num_classes = len(dataset.classes)
    model = SignDetectionModel(num_classes=num_classes).to(device)

    console.print(f"[blue]Model created with {num_classes} classes: {dataset.classes}[/blue]")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Tensorboard
    writer = SummaryWriter(log_dir='logs/tensorboard')

    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        console.print(f"\n[bold cyan]Epoch {epoch}/{args.epochs}[/bold cyan]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, console
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log
        console.print(f"[green]Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}[/green]")
        console.print(f"[green]Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}[/green]")

        writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
        writer.add_scalar('Train/Acc_Epoch', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': dataset.classes
            }, checkpoint_dir / 'best_model.pth')
            console.print(f"[bold green]✓ Saved best model with val_acc: {val_acc:.4f}[/bold green]")

        scheduler.step()

    writer.close()
    console.print(f"\n[bold green]Training complete! Best val accuracy: {best_val_acc:.4f}[/bold green]")


if __name__ == '__main__':
    main()
