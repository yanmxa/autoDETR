"""
DETR Training Script

This script trains the DETR (Detection Transformer) model for object detection.
Uses rich for beautiful and informative console output.
"""

import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path

from detr.data.dataset import DETRDataset, collate_fn
from detr.model import DETR
from detr.loss import DETRLoss, HungarianMatcher, compute_total_loss
from detr.utils import (
    display_training_header,
    create_training_progress,
    display_checkpoint_saved,
    display_training_complete,
    display_training_error,
    display_info_message,
    print_message,
    print_empty_line
)


def get_training_config():
    """
    Configure training parameters.

    Returns:
        dict: Training configuration with all hyperparameters

    Notes:
        - Set 'pretrained_path' to None to skip loading pretrained weights
        - Set 'checkpoint_dir' to None to skip saving checkpoints
        - Adjust 'save_interval' to control checkpoint frequency
    """
    config = {
        # Data paths
        'train_data_dir': 'data/train',
        'test_data_dir': 'data/test',

        # Model configuration
        'num_classes': 3,
        'num_queries': 100,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'nheads': 8,
        'hidden_dim': 256,
        'dropout': 0.1,

        # Training hyperparameters
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-5,

        # Loss weights
        'loss_weights': {
            'class_weighting': 1.0,
            'bbox_weighting': 5.0,
            'giou_weighting': 2.0
        },
        'eos_coef': 0.1,  # Weight for "no-object" class

        # Optimizer and scheduler
        'optimizer': 'Adam',  # 'Adam' or 'AdamW'
        'scheduler': 'CosineAnnealingWarmRestarts',
        'T_0': None,  # Will be set to len(train_dataloader) * 30 if None
        'T_mult': 2,

        # Checkpoint configuration
        'pretrained_path': None,  # Set to path string to load pretrained model, None to skip
        'checkpoint_dir': 'checkpoints',  # Directory to save checkpoints, None to skip
        'save_interval': 10,  # Save checkpoint every N epochs

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    return config




def train_one_epoch(model, dataloader, criterion, optimizer, device, progress, task_id):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss_dict = criterion(predictions, targets)
        total_loss = compute_total_loss(loss_dict, criterion.weight_dict)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += total_loss.item()

        # Update progress
        progress.update(task_id, advance=1)

    return epoch_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    epoch_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in dataloader:
            # Move to device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss_dict = criterion(predictions, targets)
            total_loss = compute_total_loss(loss_dict, criterion.weight_dict)

            # Accumulate loss
            epoch_loss += total_loss.item()

    return epoch_loss / num_batches


def save_checkpoint(model, epoch, save_dir):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"epoch_{epoch}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    display_checkpoint_saved(str(checkpoint_path))


def main():
    """Main training function."""
    # Get configuration
    config = get_training_config()

    # Display training header using centralized function
    display_training_header(config)

    # Create datasets
    print_message("Loading datasets...", "bold")
    train_dataset = DETRDataset(config['train_data_dir'], train=True)
    test_dataset = DETRDataset(config['test_data_dir'], train=False)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        collate_fn=collate_fn,
        drop_last=True
    )

    print_message(f"[green]✓[/green] Train batches: [yellow]{len(train_dataloader)}[/yellow]")
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
        verbose=True
    )

    # Load pretrained weights if specified
    if config['pretrained_path'] is not None:
        print_message(f"[bold]Loading pretrained weights from:[/bold] [cyan]{config['pretrained_path']}[/cyan]")
        try:
            model.load_pretrained(config['pretrained_path'])
        except Exception as e:
            display_info_message(f"Failed to load pretrained weights: {str(e)}", style="red")
            display_info_message("Continuing with random initialization...", style="yellow")
    else:
        display_info_message("No pretrained weights specified. Using random initialization.", style="yellow")

    print_empty_line()

    # Move model to device
    device = torch.device(config['device'])
    model = model.to(device)
    print_message(f"[green]✓[/green] Model moved to device: [yellow]{device}[/yellow]")
    print_empty_line()

    # Create optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # Create scheduler
    T_0 = config['T_0'] if config['T_0'] is not None else len(train_dataloader) * 30
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=config['T_mult']
    )

    # Create loss criterion
    matcher = HungarianMatcher(config['loss_weights'])
    criterion = DETRLoss(
        num_classes=config['num_classes'],
        matcher=matcher,
        weight_dict=config['loss_weights'],
        eos_coef=config['eos_coef']
    )

    # Training loop
    print_message("[bold green]Starting Training[/bold green]")
    print_empty_line()

    best_test_loss = float('inf')

    with create_training_progress() as progress:
        for epoch in range(config['epochs']):
            # Create progress task for this epoch
            task = progress.add_task(
                "",
                total=len(train_dataloader),
                epoch_info=f"Epoch {epoch+1}/{config['epochs']}",
                train_loss=0.0,
                test_loss=0.0
            )

            # Train one epoch
            try:
                train_loss = train_one_epoch(
                    model, train_dataloader, criterion, optimizer, device, progress, task
                )

                # Evaluate
                test_loss = evaluate(model, test_dataloader, criterion, device)

                # Update progress with final losses
                progress.update(
                    task,
                    epoch_info=f"Epoch {epoch+1}/{config['epochs']}",
                    train_loss=train_loss,
                    test_loss=test_loss
                )

                # Step scheduler
                scheduler.step()

                # Save best model
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    if config['checkpoint_dir'] is not None:
                        save_checkpoint(model, "best", config['checkpoint_dir'])

                # Save periodic checkpoint
                if config['checkpoint_dir'] is not None:
                    if (epoch + 1) % config['save_interval'] == 0:
                        save_checkpoint(model, epoch + 1, config['checkpoint_dir'])

            except Exception as e:
                print_empty_line()
                display_training_error(epoch + 1, e)
                import traceback
                print_message(traceback.format_exc())
                sys.exit(1)

    # Save final model
    print_empty_line()
    if config['checkpoint_dir'] is not None:
        save_checkpoint(model, "final", config['checkpoint_dir'])

    # Display training complete message
    display_training_complete(best_test_loss)


if __name__ == '__main__':
    main()
