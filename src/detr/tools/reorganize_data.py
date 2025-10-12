#!/usr/bin/env python3
"""
Data Reorganization Script

Combines train and test data, then redistributes them with balanced class distribution.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
from detr.utils import print_message, print_empty_line
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import box


def collect_all_data(data_dirs):
    """
    Collect all images and labels from multiple directories.

    Args:
        data_dirs: List of data directories to collect from

    Returns:
        Dict mapping class names to list of (image_path, label_path) tuples
    """
    data_by_class = defaultdict(list)

    for data_dir in data_dirs:
        images_dir = Path(data_dir) / 'images'
        labels_dir = Path(data_dir) / 'labels'

        if not images_dir.exists():
            print_message(f"[yellow]⚠[/yellow] Directory not found: {images_dir}")
            continue

        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        for img_path in image_files:
            # Extract class name from filename
            # Expected format: *-{class}-*.jpg
            parts = img_path.stem.split('-')
            if len(parts) >= 2:
                class_name = None
                # Find class name in filename
                for part in parts:
                    if part in ['one', 'two', 'three']:
                        class_name = part
                        break

                if class_name:
                    # Find corresponding label file
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        data_by_class[class_name].append((img_path, label_path))
                    else:
                        print_message(f"[yellow]⚠[/yellow] Label not found: {label_path}")

    return data_by_class


def split_data(data_by_class, train_ratio=0.8, random_seed=42):
    """
    Split data into train and test sets with balanced distribution.

    Args:
        data_by_class: Dict mapping class names to list of (image, label) tuples
        train_ratio: Ratio of training data (default: 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data) where each is a list of (image, label, class) tuples
    """
    random.seed(random_seed)

    train_data = []
    test_data = []

    for class_name, samples in data_by_class.items():
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Calculate split point
        n_train = int(len(shuffled) * train_ratio)

        # Split
        for i, (img_path, label_path) in enumerate(shuffled):
            if i < n_train:
                train_data.append((img_path, label_path, class_name))
            else:
                test_data.append((img_path, label_path, class_name))

    # Shuffle again to mix classes
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def copy_data(data_list, target_dir):
    """
    Copy images and labels to target directory.

    Args:
        data_list: List of (image_path, label_path, class_name) tuples
        target_dir: Target directory (will create images/ and labels/ subdirs)
    """
    target_dir = Path(target_dir)
    images_dir = target_dir / 'images'
    labels_dir = target_dir / 'labels'

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img_path, label_path, class_name in data_list:
        # Copy image
        shutil.copy2(img_path, images_dir / img_path.name)
        # Copy label
        shutil.copy2(label_path, labels_dir / label_path.name)


def display_statistics(data_by_class, train_data, test_data):
    """Display data distribution statistics."""
    console = Console()

    # Original distribution
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Data Reorganization Statistics[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Original data table
    orig_table = Table(
        title="📊 Original Data Distribution",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    orig_table.add_column("Class", style="cyan", no_wrap=True)
    orig_table.add_column("Count", style="yellow", justify="right")

    total_orig = 0
    for class_name in sorted(data_by_class.keys()):
        count = len(data_by_class[class_name])
        total_orig += count
        orig_table.add_row(class_name, str(count))

    orig_table.add_row("[bold]Total[/bold]", f"[bold]{total_orig}[/bold]")
    console.print(orig_table)

    # New distribution table
    train_by_class = defaultdict(int)
    test_by_class = defaultdict(int)

    for _, _, class_name in train_data:
        train_by_class[class_name] += 1

    for _, _, class_name in test_data:
        test_by_class[class_name] += 1

    new_table = Table(
        title="📈 New Data Distribution (Train/Test Split)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green"
    )
    new_table.add_column("Class", style="cyan", no_wrap=True)
    new_table.add_column("Train", style="green", justify="right")
    new_table.add_column("Test", style="blue", justify="right")
    new_table.add_column("Total", style="yellow", justify="right")

    total_train = 0
    total_test = 0

    for class_name in sorted(data_by_class.keys()):
        train_count = train_by_class[class_name]
        test_count = test_by_class[class_name]
        total_count = train_count + test_count

        total_train += train_count
        total_test += test_count

        new_table.add_row(
            class_name,
            str(train_count),
            str(test_count),
            str(total_count)
        )

    new_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_train}[/bold]",
        f"[bold]{total_test}[/bold]",
        f"[bold]{total_train + total_test}[/bold]"
    )

    console.print(new_table)

    # Split ratio
    ratio = total_train / (total_train + total_test) * 100
    console.print(
        f"\n[cyan]ℹ[/cyan] Train/Test ratio: "
        f"[green]{ratio:.1f}%[/green] / [blue]{100-ratio:.1f}%[/blue]"
    )
    console.print()


def main():
    """Main function."""
    print_message("\n[bold cyan]Data Reorganization Tool[/bold cyan]\n")

    # Configuration
    source_dirs = ['data/train', 'data/test']
    target_train_dir = 'data_new/train'
    target_test_dir = 'data_new/test'
    train_ratio = 0.75  # 75% train, 25% test
    random_seed = 42

    print_message(f"[bold]Configuration:[/bold]")
    print_message(f"  Source directories: [cyan]{', '.join(source_dirs)}[/cyan]")
    print_message(f"  Target train dir: [cyan]{target_train_dir}[/cyan]")
    print_message(f"  Target test dir: [cyan]{target_test_dir}[/cyan]")
    print_message(f"  Train ratio: [cyan]{train_ratio*100:.0f}%[/cyan]")
    print_message(f"  Random seed: [cyan]{random_seed}[/cyan]")
    print_empty_line()

    # Step 1: Collect all data
    print_message("[bold]Step 1:[/bold] Collecting data from source directories...")
    data_by_class = collect_all_data(source_dirs)

    if not data_by_class:
        print_message("[red]✗[/red] No data found in source directories!")
        return

    total_samples = sum(len(samples) for samples in data_by_class.values())
    print_message(f"[green]✓[/green] Collected [yellow]{total_samples}[/yellow] samples across [yellow]{len(data_by_class)}[/yellow] classes")
    print_empty_line()

    # Step 2: Split data
    print_message("[bold]Step 2:[/bold] Splitting data into train and test sets...")
    train_data, test_data = split_data(data_by_class, train_ratio, random_seed)
    print_message(f"[green]✓[/green] Train: [yellow]{len(train_data)}[/yellow] samples")
    print_message(f"[green]✓[/green] Test: [yellow]{len(test_data)}[/yellow] samples")
    print_empty_line()

    # Step 3: Display statistics
    display_statistics(data_by_class, train_data, test_data)

    # Step 4: Copy to new directories
    print_message("[bold]Step 3:[/bold] Copying data to new directories...")

    # Backup old directories if they exist
    for old_dir in [target_train_dir, target_test_dir]:
        old_path = Path(old_dir)
        if old_path.exists():
            backup_path = Path(str(old_path) + '_backup')
            if backup_path.exists():
                shutil.rmtree(backup_path)
            print_message(f"[yellow]⚠[/yellow] Backing up existing directory: {old_dir} → {backup_path}")
            shutil.move(str(old_path), str(backup_path))

    copy_data(train_data, target_train_dir)
    print_message(f"[green]✓[/green] Copied train data to: [cyan]{target_train_dir}[/cyan]")

    copy_data(test_data, target_test_dir)
    print_message(f"[green]✓[/green] Copied test data to: [cyan]{target_test_dir}[/cyan]")
    print_empty_line()

    # Step 5: Done
    print_message("[bold green]✓ Data reorganization complete![/bold green]")
    print_message("\n[bold]Next steps:[/bold]")
    print_message("  1. Review the new data in [cyan]data_new/[/cyan]")
    print_message("  2. Update training config to use [cyan]data_new/train[/cyan] and [cyan]data_new/test[/cyan]")
    print_message("  3. Run training: [cyan]detr-train[/cyan]")
    print_empty_line()


if __name__ == '__main__':
    main()
