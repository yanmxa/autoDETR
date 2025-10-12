#!/usr/bin/env python3
"""
Training History Analysis Tool

Analyzes training history from CSV file and provides insights.
"""

import csv
import sys
from pathlib import Path
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import box


def load_training_history(csv_path='training_history.csv'):
    """
    Load training history from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (epochs, train_losses, test_losses)
    """
    epochs = []
    train_losses = []
    test_losses = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            test_losses.append(float(row['test_loss']))

    return epochs, train_losses, test_losses


def analyze_training(epochs, train_losses, test_losses):
    """
    Analyze training performance and provide insights.

    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        test_losses: List of test losses

    Returns:
        Dict of analysis results
    """
    if not epochs:
        return {}

    # Basic statistics
    min_train_loss = min(train_losses)
    min_test_loss = min(test_losses)
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]

    min_train_epoch = epochs[train_losses.index(min_train_loss)]
    min_test_epoch = epochs[test_losses.index(min_test_loss)]

    # Calculate loss reduction
    initial_train_loss = train_losses[0]
    initial_test_loss = test_losses[0]
    train_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
    test_reduction = ((initial_test_loss - final_test_loss) / initial_test_loss) * 100

    # Detect overfitting (last 10 epochs)
    if len(epochs) >= 10:
        recent_train = train_losses[-10:]
        recent_test = test_losses[-10:]
        train_trend = recent_train[-1] - recent_train[0]
        test_trend = recent_test[-1] - recent_test[0]

        is_overfitting = train_trend < 0 and test_trend > 0
        is_underfitting = train_trend > 0 and test_trend > 0
    else:
        is_overfitting = False
        is_underfitting = False
        train_trend = 0
        test_trend = 0

    # Gap between train and test
    final_gap = final_test_loss - final_train_loss
    gap_percentage = (final_gap / final_train_loss) * 100

    return {
        'total_epochs': len(epochs),
        'initial_train_loss': initial_train_loss,
        'initial_test_loss': initial_test_loss,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'min_train_loss': min_train_loss,
        'min_test_loss': min_test_loss,
        'min_train_epoch': min_train_epoch,
        'min_test_epoch': min_test_epoch,
        'train_reduction': train_reduction,
        'test_reduction': test_reduction,
        'final_gap': final_gap,
        'gap_percentage': gap_percentage,
        'is_overfitting': is_overfitting,
        'is_underfitting': is_underfitting,
        'train_trend': train_trend,
        'test_trend': test_trend,
    }


def display_analysis(analysis):
    """Display analysis results."""
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Training History Analysis[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Basic statistics table
    stats_table = Table(
        title="📊 Training Statistics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="yellow", justify="right")

    stats_table.add_row("Total Epochs", str(analysis['total_epochs']))
    stats_table.add_row("Initial Train Loss", f"{analysis['initial_train_loss']:.4f}")
    stats_table.add_row("Initial Test Loss", f"{analysis['initial_test_loss']:.4f}")
    stats_table.add_row("Final Train Loss", f"{analysis['final_train_loss']:.4f}")
    stats_table.add_row("Final Test Loss", f"{analysis['final_test_loss']:.4f}")
    stats_table.add_row("Best Train Loss", f"{analysis['min_train_loss']:.4f} (epoch {analysis['min_train_epoch']})")
    stats_table.add_row("Best Test Loss", f"{analysis['min_test_loss']:.4f} (epoch {analysis['min_test_epoch']})")

    console.print(stats_table)

    # Performance table
    perf_table = Table(
        title="📈 Performance Improvement",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green"
    )
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="yellow", justify="right")

    # Color code based on performance
    train_color = "green" if analysis['train_reduction'] > 20 else "yellow" if analysis['train_reduction'] > 10 else "red"
    test_color = "green" if analysis['test_reduction'] > 20 else "yellow" if analysis['test_reduction'] > 10 else "red"

    perf_table.add_row("Train Loss Reduction", f"[{train_color}]{analysis['train_reduction']:.1f}%[/{train_color}]")
    perf_table.add_row("Test Loss Reduction", f"[{test_color}]{analysis['test_reduction']:.1f}%[/{test_color}]")
    perf_table.add_row("Train-Test Gap", f"{analysis['final_gap']:.4f}")
    perf_table.add_row("Gap Percentage", f"{analysis['gap_percentage']:.1f}%")

    console.print(perf_table)

    # Diagnosis
    console.print()
    console.print(Panel(
        get_diagnosis_text(analysis),
        title="🔍 Diagnosis",
        border_style="blue",
        box=box.ROUNDED
    ))

    console.print()


def get_diagnosis_text(analysis):
    """Generate diagnosis text based on analysis."""
    lines = []

    # Check overall performance
    if analysis['final_test_loss'] < 3.0:
        lines.append("[bold green]✓ Excellent:[/bold green] Model has converged well!")
    elif analysis['final_test_loss'] < 5.0:
        lines.append("[bold yellow]⚠ Good:[/bold yellow] Model shows decent performance, but could improve.")
    else:
        lines.append("[bold red]✗ Poor:[/bold red] Model needs more training or configuration changes.")

    # Check overfitting/underfitting
    if analysis['is_overfitting']:
        lines.append("[bold red]⚠ Overfitting detected:[/bold red] Test loss increasing while train loss decreasing.")
        lines.append("  [dim]→ Suggestions: Add dropout, collect more data, reduce model complexity[/dim]")
    elif analysis['is_underfitting']:
        lines.append("[bold yellow]⚠ Underfitting:[/bold yellow] Both losses increasing or stagnating.")
        lines.append("  [dim]→ Suggestions: Increase model layers, increase learning rate, train longer[/dim]")
    else:
        lines.append("[bold green]✓ Training healthy:[/bold green] No overfitting/underfitting detected.")

    # Check train-test gap
    if analysis['gap_percentage'] > 50:
        lines.append(f"[bold red]⚠ Large gap:[/bold red] {analysis['gap_percentage']:.1f}% gap between train and test.")
        lines.append("  [dim]→ Suggests overfitting or train/test data mismatch[/dim]")
    elif analysis['gap_percentage'] > 20:
        lines.append(f"[bold yellow]⚠ Moderate gap:[/bold yellow] {analysis['gap_percentage']:.1f}% gap - acceptable but could improve.")
    else:
        lines.append(f"[bold green]✓ Good generalization:[/bold green] Small gap ({analysis['gap_percentage']:.1f}%).")

    # Check improvement
    if analysis['test_reduction'] > 30:
        lines.append(f"[bold green]✓ Strong improvement:[/bold green] {analysis['test_reduction']:.1f}% test loss reduction!")
    elif analysis['test_reduction'] > 15:
        lines.append(f"[bold yellow]⚠ Moderate improvement:[/bold yellow] {analysis['test_reduction']:.1f}% reduction.")
    else:
        lines.append(f"[bold red]✗ Weak improvement:[/bold red] Only {analysis['test_reduction']:.1f}% reduction.")
        lines.append("  [dim]→ Consider: Higher learning rate, more epochs, better architecture[/dim]")

    return "\n".join(lines)


def main():
    """Main function."""
    console = Console()

    # Check if CSV file exists
    csv_path = Path('training_history.csv')
    if not csv_path.exists():
        console.print("[red]✗[/red] training_history.csv not found!")
        console.print("[yellow]ℹ[/yellow] Run training first: [cyan]detr-train[/cyan]")
        sys.exit(1)

    console.print(f"\n[bold]Loading training history from:[/bold] [cyan]{csv_path}[/cyan]")

    # Load and analyze
    try:
        epochs, train_losses, test_losses = load_training_history(csv_path)
        analysis = analyze_training(epochs, train_losses, test_losses)

        # Display results
        display_analysis(analysis)

        # Show first and last few epochs
        console.print(Panel(
            f"[bold]First 5 epochs:[/bold]\n" +
            "\n".join([f"Epoch {e}: Train={tl:.4f}, Test={tel:.4f}"
                      for e, tl, tel in zip(epochs[:5], train_losses[:5], test_losses[:5])]) +
            f"\n\n[bold]Last 5 epochs:[/bold]\n" +
            "\n".join([f"Epoch {e}: Train={tl:.4f}, Test={tel:.4f}"
                      for e, tl, tel in zip(epochs[-5:], train_losses[-5:], test_losses[-5:])]),
            title="📋 Epoch Details",
            border_style="cyan",
            box=box.ROUNDED
        ))
        console.print()

    except Exception as e:
        console.print(f"[red]✗[/red] Error analyzing training history: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
