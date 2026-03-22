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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detr.data.dataset import DETRDataset, collate_fn
from detr.model import DETR
from detr.loss import DETRLoss, HungarianMatcher, compute_total_loss
from detr.config import get_training_config
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




def train_one_epoch(model, dataloader, criterion, optimizer, device, progress, task_id, config, epoch_num):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = len(dataloader)
    valid_batches = 0
    nan_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss_dict = criterion(predictions, targets)
        total_loss = compute_total_loss(loss_dict, criterion.weight_dict)

        # Check for NaN or Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_batches += 1
            print_message(f"\n[red]✗[/red] Epoch {epoch_num+1}, Batch {batch_idx+1}: NaN/Inf!")
            progress.update(task_id, advance=1)
            continue

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.get('grad_clip_max_norm', 1.0)
        )

        optimizer.step()

        # Accumulate loss
        epoch_loss += total_loss.item()
        valid_batches += 1

        # Update progress bar (don't update test_loss during training)
        current_avg = epoch_loss / valid_batches
        progress.update(
            task_id,
            advance=1,
            epoch_info=f"Epoch {epoch_num+1}/{config['epochs']}",
            train_loss=current_avg,
            test_loss=None  # Keep test_loss undefined during training
        )

    # Summary
    if nan_batches > 0:
        print_message(f"\n[yellow]⚠[/yellow] Epoch {epoch_num+1}: {nan_batches}/{num_batches} batches had NaN/Inf")

    if valid_batches == 0:
        print_message(f"\n[red]✗[/red] Epoch {epoch_num+1}: All batches failed!")
        return float('inf')

    return epoch_loss / valid_batches


def evaluate(model, dataloader, criterion, device, progress, task_id, train_loss):
    """Evaluate model on test set."""
    model.eval()
    epoch_loss = 0.0
    valid_batches = 0
    nan_batches = 0

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

            # Check for NaN or Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_batches += 1
                continue

            # Accumulate loss
            epoch_loss += total_loss.item()
            valid_batches += 1

    # Handle errors
    if nan_batches > 0:
        print_message(f"\n[yellow]⚠[/yellow] Test: {nan_batches} batches had NaN/Inf")

    if valid_batches == 0:
        print_message(f"\n[red]✗[/red] All test batches failed!")
        test_loss = float('inf')
    else:
        test_loss = epoch_loss / valid_batches

    # Don't update progress here - just return the test loss
    # The main loop will print the completed epoch info

    return test_loss


def save_checkpoint(model, epoch, save_dir):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"epoch_{epoch}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Don't print here - caller will print using progress.console


def save_loss_history(epochs_list, train_losses, test_losses, save_path='training_history.csv'):
    """
    Save training history to CSV file.

    Args:
        epochs_list: List of epoch numbers
        train_losses: List of training losses
        test_losses: List of test losses
        save_path: Path to save the CSV file
    """
    import csv

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])
        for epoch, train_loss, test_loss in zip(epochs_list, train_losses, test_losses):
            writer.writerow([epoch, f'{train_loss:.6f}', f'{test_loss:.6f}'])

    print_message(f"[green]✓[/green] Training history saved to: [cyan]{save_path}[/cyan]")


def plot_loss_curves(epochs_list, train_losses, test_losses, save_path='training_curves.png'):
    """
    Plot training and test loss curves.

    Args:
        epochs_list: List of epoch numbers
        train_losses: List of training losses
        test_losses: List of test losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot losses
    plt.plot(epochs_list, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    plt.plot(epochs_list, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=4)

    # Formatting
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Set x-axis to show integer epochs
    plt.xticks(range(0, max(epochs_list) + 1, max(1, max(epochs_list) // 10)))

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print_message(f"[green]✓[/green] Loss curves saved to: [cyan]{save_path}[/cyan]")

    plt.close()


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

    # Create scheduler (supports both CosineAnnealing and ReduceLROnPlateau)
    if config['scheduler'] == 'CosineAnnealingWarmRestarts':
        T_0 = config.get('T_0') if config.get('T_0') is not None else len(train_dataloader) * 30
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=config.get('T_mult', 2)
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('patience', 10),
            min_lr=config.get('min_lr', 1e-6)
            # verbose parameter removed - not supported in all PyTorch versions
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

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

    # Lists to store loss history
    train_losses = []
    test_losses = []
    epochs_list = []

    with create_training_progress() as progress:
        # Use only one progress task that gets updated each epoch
        task = progress.add_task(
            "",
            total=len(train_dataloader),
            epoch_info=f"Epoch 1/{config['epochs']}",
            train_loss=0.0,
            test_loss=None
        )

        for epoch in range(config['epochs']):
            # Reset the task for new epoch
            progress.reset(
                task,
                total=len(train_dataloader),
                epoch_info=f"Epoch {epoch+1}/{config['epochs']}",
                train_loss=0.0,
                test_loss=None
            )

            # Train one epoch
            try:
                train_loss = train_one_epoch(
                    model, train_dataloader, criterion, optimizer, device,
                    progress, task, config, epoch
                )

                # Evaluate (pass progress to prevent UI freeze)
                test_loss = evaluate(model, test_dataloader, criterion, device, progress, task, train_loss)

                # Build checkpoint info
                checkpoint_info = ""
                save_best = False
                save_periodic = False

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    save_best = True

                if config['checkpoint_dir'] is not None and (epoch + 1) % config['save_interval'] == 0:
                    save_periodic = True

                # Build checkpoint message
                if save_best and save_periodic:
                    checkpoint_info = f" [green]✓[/green] [dim]Saved: best, epoch_{epoch+1}[/dim]"
                elif save_best:
                    checkpoint_info = f" [green]✓[/green] [dim]Saved: best[/dim]"
                elif save_periodic:
                    checkpoint_info = f" [green]✓[/green] [dim]Saved: epoch_{epoch+1}[/dim]"

                # Print completed epoch info using progress.console to avoid flickering
                progress.console.print(
                    f"  [blue]Epoch {epoch+1}/{config['epochs']}[/blue] "
                    f"[cyan]Train Loss: {train_loss:.5f}[/cyan] • "
                    f"[magenta]Test Loss: {test_loss:.5f}[/magenta]"
                    f"{checkpoint_info}"
                )

                # Record loss history
                epochs_list.append(epoch + 1)
                train_losses.append(train_loss)
                test_losses.append(test_loss)

                # Step scheduler (different for ReduceLROnPlateau)
                if config['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(test_loss)  # Needs metric
                else:
                    scheduler.step()  # CosineAnnealing doesn't need metric

                # Actually save checkpoints
                if config['checkpoint_dir'] is not None:
                    if save_best:
                        save_checkpoint(model, "best", config['checkpoint_dir'])
                    if save_periodic:
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

    # Save training history to CSV
    print_empty_line()
    print_message("[bold]Saving training history...[/bold]")
    save_loss_history(epochs_list, train_losses, test_losses, 'training_history.csv')

    # Plot and save loss curves
    print_message("[bold]Generating loss curves...[/bold]")
    plot_loss_curves(epochs_list, train_losses, test_losses, 'training_curves.png')

    # Run validation (metrics only, no image — use autodetr-val --tag <name> to save image)
    print_empty_line()
    print_message("[bold]Running validation...[/bold]")
    from detr.validate import run_validation
    run_validation()  # tag=None → metrics only, no visualization saved


if __name__ == '__main__':
    main()
