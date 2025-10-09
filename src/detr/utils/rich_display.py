"""
Rich display utilities for DETR project.

Centralized rich formatting functions for consistent, beautiful console output
across dataset, model, loss, and training modules.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich import box


# ============================================================================
# Dataset Display Functions
# ============================================================================

def display_dataset_info(data_dir: str, train: bool, num_samples: int,
                        image_size: int, images_path: str, labels_path: str):
    """
    Display dataset initialization information.

    Args:
        data_dir: Root data directory
        train: Whether this is training dataset
        num_samples: Total number of samples
        image_size: Target image size
        images_path: Path to images directory
        labels_path: Path to labels directory
    """
    console = Console()

    # Dataset statistics table
    table = Table(
        title="📊 DETR Dataset Statistics",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow")

    mode_emoji = "🏋️" if train else "🧪"
    mode_text = f"{mode_emoji} {'Training' if train else 'Validation/Test'}"

    table.add_row("Data Path", data_dir)
    table.add_row("Mode", mode_text)
    table.add_row("Total Samples", str(num_samples))
    table.add_row("Image Size", f"{image_size}×{image_size}")
    table.add_row("Images Path", images_path)
    table.add_row("Labels Path", labels_path)

    console.print(table)

    # Data transforms panel
    if train:
        aug_list = [
            "• Resize to 500×500",
            "• Random Crop to 224×224 (p=0.33)",
            "• Resize to final size",
            "• Horizontal Flip (p=0.5)",
            "• Color Jitter (p=0.5)",
            "• Normalize (ImageNet stats)",
            "• Convert to Tensor"
        ]
    else:
        aug_list = [
            "• Resize to target size",
            "• Normalize (ImageNet stats)",
            "• Convert to Tensor"
        ]

    aug_text = "\n".join(aug_list)
    panel = Panel(
        aug_text,
        title="🔄 Data Transforms",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


# ============================================================================
# Model Display Functions
# ============================================================================

def display_model_info(num_classes: int, hidden_dim: int, nheads: int,
                      num_encoder_layers: int, num_decoder_layers: int,
                      num_queries: int, dropout: float, total_params: int,
                      trainable_params: int):
    """
    Display model architecture information.

    Args:
        num_classes: Number of object classes
        hidden_dim: Transformer hidden dimension
        nheads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_queries: Number of object queries
        dropout: Dropout rate
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]DETR Model Initialized[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Architecture table
    arch_table = Table(
        title="🏗️  Model Architecture",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    arch_table.add_column("Component", style="cyan", no_wrap=True)
    arch_table.add_column("Configuration", style="yellow")

    arch_table.add_row("Backbone", "ResNet-50 (ImageNet pretrained)")
    arch_table.add_row("Hidden Dimension", str(hidden_dim))
    arch_table.add_row("Attention Heads", str(nheads))
    arch_table.add_row("Encoder Layers", str(num_encoder_layers))
    arch_table.add_row("Decoder Layers", str(num_decoder_layers))
    arch_table.add_row("Dropout Rate", f"{dropout:.2f}")

    console.print(arch_table)

    # Detection configuration
    det_table = Table(
        title="🎯 Detection Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green"
    )
    det_table.add_column("Parameter", style="cyan")
    det_table.add_column("Value", style="yellow", justify="right")

    det_table.add_row("Number of Classes", str(num_classes))
    det_table.add_row("Object Queries", str(num_queries))
    det_table.add_row("Output Classes", f"{num_classes + 1} (+ background)")

    console.print(det_table)

    # Parameter count
    param_table = Table(
        title="📊 Parameters",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue"
    )
    param_table.add_column("Type", style="cyan")
    param_table.add_column("Count", style="yellow", justify="right")

    param_table.add_row("Total Parameters", f"{total_params:,}")
    param_table.add_row("Trainable Parameters", f"{trainable_params:,}")
    param_table.add_row(
        "Model Size (FP32)",
        f"{total_params * 4 / (1024**2):.2f} MB"
    )

    console.print(param_table)
    console.print()


def display_checkpoint_loaded(checkpoint_path: str, success: bool = True, error_msg: str = None):
    """
    Display checkpoint loading status.

    Args:
        checkpoint_path: Path to checkpoint file
        success: Whether loading was successful
        error_msg: Error message if failed
    """
    console = Console()
    console.print()

    if success:
        console.print(Panel(
            f"[bold green]✓[/bold green] Successfully loaded checkpoint from:\n"
            f"[cyan]{checkpoint_path}[/cyan]",
            title="✅ Checkpoint Loaded",
            border_style="green",
            box=box.ROUNDED
        ))
    else:
        console.print(Panel(
            f"[bold red]✗[/bold red] Failed to load checkpoint:\n"
            f"[red]{error_msg}[/red]\n\n"
            f"Path: [cyan]{checkpoint_path}[/cyan]",
            title="❌ Loading Failed",
            border_style="red",
            box=box.ROUNDED
        ))

    console.print()


# ============================================================================
# Training Display Functions
# ============================================================================

def display_training_header(config: dict):
    """
    Display training configuration header.

    Args:
        config: Training configuration dictionary
    """
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]DETR Object Detection Training[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Model configuration table
    model_table = Table(
        title="🏗️  Model Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    model_table.add_column("Parameter", style="cyan", no_wrap=True)
    model_table.add_column("Value", style="yellow")

    model_table.add_row("Number of Classes", str(config['num_classes']))
    model_table.add_row("Object Queries", str(config['num_queries']))
    model_table.add_row("Encoder Layers", str(config['num_encoder_layers']))
    model_table.add_row("Decoder Layers", str(config['num_decoder_layers']))
    model_table.add_row("Attention Heads", str(config['nheads']))
    model_table.add_row("Hidden Dimension", str(config['hidden_dim']))
    model_table.add_row("Dropout", f"{config['dropout']:.2f}")

    console.print(model_table)

    # Training configuration table
    train_table = Table(
        title="🏋️  Training Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green"
    )
    train_table.add_column("Parameter", style="cyan")
    train_table.add_column("Value", style="yellow", justify="right")

    train_table.add_row("Total Epochs", str(config['epochs']))
    train_table.add_row("Batch Size", str(config['batch_size']))
    train_table.add_row("Learning Rate", f"{config['learning_rate']:.0e}")
    train_table.add_row("Optimizer", config['optimizer'])
    train_table.add_row("Scheduler", config['scheduler'])
    train_table.add_row("Device", config['device'].upper())

    console.print(train_table)

    # Loss configuration table
    loss_table = Table(
        title="⚖️  Loss Weights",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue"
    )
    loss_table.add_column("Loss Component", style="cyan")
    loss_table.add_column("Weight", style="yellow", justify="right")

    loss_table.add_row("Classification", f"{config['loss_weights']['class_weighting']:.1f}")
    loss_table.add_row("BBox Regression (L1)", f"{config['loss_weights']['bbox_weighting']:.1f}")
    loss_table.add_row("GIoU", f"{config['loss_weights']['giou_weighting']:.1f}")
    loss_table.add_row("EOS Coefficient", f"{config['eos_coef']:.2f}")

    console.print(loss_table)
    console.print()


def create_training_progress():
    """
    Create a rich progress bar for training.

    Returns:
        Progress: Configured progress bar object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[epoch_info]}", justify="left"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]Train Loss: {task.fields[train_loss]:.5f}"),
        TextColumn("•"),
        TextColumn("[magenta]Test Loss: {task.fields[test_loss]:.5f}"),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=False
    )


def display_checkpoint_saved(checkpoint_path: str):
    """
    Display checkpoint saved message.

    Args:
        checkpoint_path: Path to saved checkpoint
    """
    console = Console()
    console.print(f"[green]✓[/green] Checkpoint saved: [cyan]{checkpoint_path}[/cyan]")


def display_training_complete(best_test_loss: float):
    """
    Display training completion message.

    Args:
        best_test_loss: Best test loss achieved during training
    """
    console = Console()
    console.print()
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n\n"
        f"[cyan]Best Test Loss:[/cyan] [yellow]{best_test_loss:.5f}[/yellow]",
        border_style="green",
        box=box.DOUBLE
    ))


def display_training_error(epoch: int, error: Exception):
    """
    Display training error message.

    Args:
        epoch: Epoch number where error occurred
        error: Exception object
    """
    console = Console()
    console.print()
    console.print(Panel(
        f"[red bold]Training Error at Epoch {epoch}[/red bold]\n\n"
        f"[red]{str(error)}[/red]",
        title="❌ Error",
        border_style="red",
        box=box.ROUNDED
    ))


def display_info_message(message: str, style: str = "yellow"):
    """
    Display informational message.

    Args:
        message: Message to display
        style: Color style (yellow, green, cyan, red)
    """
    console = Console()
    icon = {
        'yellow': '⚠',
        'green': '✓',
        'cyan': 'ℹ',
        'red': '✗'
    }.get(style, 'ℹ')

    console.print(f"[{style}]{icon}[/{style}] {message}")


# ============================================================================
# Image Capture Display Functions
# ============================================================================

def display_capture_banner():
    """Display image capture system banner."""
    console = Console()
    banner = Panel.fit(
        "[bold cyan]IMAGE CAPTURE SYSTEM v1.0[/bold cyan]\n"
        "[dim]Sign Language Dataset Collection[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    )
    console.print(banner)


def display_capture_session_info(classes: list, num_images: int, sleep_time: int):
    """
    Display capture session configuration.

    Args:
        classes: List of class names
        num_images: Number of images per class
        sleep_time: Interval between captures in seconds
    """
    console = Console()
    table = Table(
        title="Capture Session Configuration",
        box=box.ROUNDED,
        border_style="cyan"
    )
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")

    table.add_row("Classes", ", ".join(classes))
    table.add_row("Images per class", str(num_images))
    table.add_row("Interval", f"{sleep_time}s")
    table.add_row("Total images", str(num_images * len(classes)))

    console.print()
    console.print(table)
    console.print()


def display_capture_session_summary(total_captured: int, num_classes: int):
    """
    Display capture session completion summary.

    Args:
        total_captured: Total number of images captured
        num_classes: Number of classes processed
    """
    console = Console()
    summary = Panel(
        f"[bold green]✓[/bold green] Total images captured: [bold]{total_captured}[/bold]\n"
        f"[bold green]✓[/bold green] Classes processed: [bold]{num_classes}[/bold]",
        title="[bold]Session Complete[/bold]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print()
    console.print(summary)


def create_capture_progress():
    """
    Create a rich progress bar for image capture.

    Returns:
        Progress: Configured progress bar object
    """
    from rich.progress import TimeElapsedColumn

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn()
    )
