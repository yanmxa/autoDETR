#!/usr/bin/env python3
"""
Dataset Display Tool

Displays sample images from the dataset with their labels.
Supports configurable number of samples to display.
"""

import cv2
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from detr.utils import print_message
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def load_label(label_path):
    """
    Load label from text file.

    Args:
        label_path: Path to label file

    Returns:
        List of label entries (class_id, x_center, y_center, width, height)
    """
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                labels.append((int(class_id), x_center, y_center, width, height))
    return labels


def draw_labels_on_image(image, labels, class_names=None):
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Image array (numpy)
        labels: List of (class_id, x_center, y_center, width, height) tuples
        class_names: Optional dict mapping class_id to class name

    Returns:
        Image with drawn annotations
    """
    img_height, img_width = image.shape[:2]
    annotated = image.copy()

    # Define colors for different classes
    colors = [
        (0, 255, 0),    # Green for class 0
        (255, 0, 0),    # Blue for class 1
        (0, 0, 255),    # Red for class 2
    ]

    for class_id, x_center, y_center, width, height in labels:
        # Convert normalized coordinates to pixel coordinates
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        box_width = int(width * img_width)
        box_height = int(height * img_height)

        # Calculate top-left corner
        x1 = int(x_center_px - box_width / 2)
        y1 = int(y_center_px - box_height / 2)
        x2 = int(x_center_px + box_width / 2)
        y2 = int(y_center_px + box_height / 2)

        # Draw rectangle
        color = colors[class_id % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label text
        label_text = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
        cv2.putText(
            annotated,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated


def collect_dataset_files(data_dir):
    """
    Collect all image and label pairs from dataset directory.

    Args:
        data_dir: Path to dataset directory (containing images/ and labels/ subdirs)

    Returns:
        Dict mapping class names to list of (image_path, label_path) tuples
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'

    if not images_dir.exists():
        print_message(f"[red]✗[/red] Images directory not found: {images_dir}")
        return {}

    if not labels_dir.exists():
        print_message(f"[red]✗[/red] Labels directory not found: {labels_dir}")
        return {}

    data_by_class = defaultdict(list)

    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    for img_path in image_files:
        # Extract class name from filename (format: *-{class}-*.jpg)
        class_name = None
        parts = img_path.stem.split('-')
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
                print_message(f"[yellow]⚠[/yellow] Label not found for: {img_path.name}")

    return data_by_class


def display_statistics(data_by_class, dataset_name):
    """
    Display dataset statistics.

    Args:
        data_by_class: Dict mapping class names to list of samples
        dataset_name: Name of the dataset (e.g., "Train", "Test")
    """
    console = Console()

    # Create statistics table
    table = Table(
        title=f"📊 {dataset_name} Dataset Statistics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Count", style="yellow", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    total = sum(len(samples) for samples in data_by_class.values())

    for class_name in sorted(data_by_class.keys()):
        count = len(data_by_class[class_name])
        percentage = (count / total * 100) if total > 0 else 0
        table.add_row(class_name, str(count), f"{percentage:.1f}%")

    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "[bold]100.0%[/bold]")

    console.print()
    console.print(table)
    console.print()


def create_image_grid(images, labels_list, class_names_list, grid_cols=3):
    """
    Create a grid layout of images.

    Args:
        images: List of images (numpy arrays)
        labels_list: List of label lists for each image
        class_names_list: List of class names for each image
        grid_cols: Number of columns in the grid

    Returns:
        Grid image combining all input images
    """
    if not images:
        return None

    # Resize all images to the same size for grid layout
    target_height = 400
    resized_images = []

    for img in images:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(img, (target_width, target_height))
        resized_images.append(resized)

    # Find max width for uniform column width
    max_width = max(img.shape[1] for img in resized_images)

    # Pad images to same width
    padded_images = []
    for img in resized_images:
        h, w = img.shape[:2]
        if w < max_width:
            padding = max_width - w
            left_pad = padding // 2
            right_pad = padding - left_pad
            padded = cv2.copyMakeBorder(img, 0, 0, left_pad, right_pad,
                                       cv2.BORDER_CONSTANT, value=(50, 50, 50))
        else:
            padded = img
        padded_images.append(padded)

    # Calculate grid dimensions
    grid_rows = (len(padded_images) + grid_cols - 1) // grid_cols

    # Create grid
    rows = []
    for row_idx in range(grid_rows):
        row_images = []
        for col_idx in range(grid_cols):
            img_idx = row_idx * grid_cols + col_idx
            if img_idx < len(padded_images):
                img = padded_images[img_idx].copy()

                # Add class name text at the top
                class_text = class_names_list[img_idx]
                cv2.putText(
                    img,
                    f"Class: {class_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                row_images.append(img)
            else:
                # Create blank placeholder
                blank = np.zeros((target_height, max_width, 3), dtype=np.uint8) + 50
                row_images.append(blank)

        row_concat = np.hstack(row_images)
        rows.append(row_concat)

    grid = np.vstack(rows)
    return grid


def display_samples(data_by_class, num_samples=5, class_names=None, random_seed=42, grid_cols=3):
    """
    Display sample images with annotations in a grid layout.

    Args:
        data_by_class: Dict mapping class names to list of (image_path, label_path) tuples
        num_samples: Number of samples to display per class
        class_names: Optional dict mapping class_id to class name
        random_seed: Random seed for reproducibility
        grid_cols: Number of columns in grid layout
    """
    random.seed(random_seed)

    # Map class names to IDs
    class_to_id = {'one': 0, 'two': 1, 'three': 2}

    if class_names is None:
        class_names = {0: 'one', 1: 'two', 2: 'three'}

    print_message(f"\n[bold cyan]Displaying {num_samples} samples per class in grid layout[/bold cyan]")
    print_message("[dim]Press any key to continue, 'q' to quit[/dim]\n")

    # Collect all samples to display
    all_images = []
    all_labels = []
    all_class_names = []

    for class_name in sorted(data_by_class.keys()):
        samples = data_by_class[class_name]

        if not samples:
            continue

        # Select random samples
        display_count = min(num_samples, len(samples))
        selected_samples = random.sample(samples, display_count)

        print_message(f"[bold magenta]Class: {class_name}[/bold magenta] ({display_count} samples)")

        for idx, (img_path, label_path) in enumerate(selected_samples, 1):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print_message(f"[red]✗[/red] Failed to load: {img_path.name}")
                continue

            # Load labels
            labels = load_label(label_path)

            # Draw annotations
            annotated_image = draw_labels_on_image(image, labels, class_names)

            all_images.append(annotated_image)
            all_labels.append(labels)
            all_class_names.append(class_name)

            print_message(
                f"  [{idx}/{display_count}] {img_path.name} - "
                f"Size: {image.shape[1]}x{image.shape[0]}, "
                f"Boxes: {len(labels)}"
            )

    if not all_images:
        print_message("[yellow]⚠[/yellow] No images to display")
        return

    # Create and display grid
    print_message(f"\n[bold]Creating grid with {len(all_images)} images ({grid_cols} columns)...[/bold]")
    grid_image = create_image_grid(all_images, all_labels, all_class_names, grid_cols)

    if grid_image is not None:
        window_name = "Dataset Samples Grid (Press any key to close, 'q' to quit)"
        cv2.imshow(window_name, grid_image)

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord('q') or key == ord('Q'):
            print_message("\n[yellow]Display closed by user[/yellow]")
        else:
            print_message("\n[green]✓[/green] Display complete!")
    else:
        print_message("[red]✗[/red] Failed to create grid")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Display dataset samples with annotations"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/test',
        help='Path to dataset directory (default: data/test)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to display per class (default: 5)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sample selection (default: 42)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Show statistics only without displaying images'
    )
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=3,
        help='Number of columns in grid layout (default: 3)'
    )

    args = parser.parse_args()

    # Display banner
    console = Console()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Dataset Display Tool[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    print_message(f"\n[bold]Configuration:[/bold]")
    print_message(f"  Dataset directory: [cyan]{args.data_dir}[/cyan]")
    print_message(f"  Samples per class: [cyan]{args.num_samples}[/cyan]")
    print_message(f"  Grid columns: [cyan]{args.grid_cols}[/cyan]")
    print_message(f"  Random seed: [cyan]{args.random_seed}[/cyan]")

    # Collect dataset files
    print_message("\n[bold]Loading dataset...[/bold]")
    data_by_class = collect_dataset_files(args.data_dir)

    if not data_by_class:
        print_message("[red]✗[/red] No data found!")
        return

    total_samples = sum(len(samples) for samples in data_by_class.values())
    print_message(f"[green]✓[/green] Loaded {total_samples} samples across {len(data_by_class)} classes")

    # Display statistics
    dataset_name = Path(args.data_dir).name.capitalize()
    display_statistics(data_by_class, dataset_name)

    # Display samples if not stats-only mode
    if not args.stats_only:
        display_samples(
            data_by_class,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            grid_cols=args.grid_cols
        )
    else:
        print_message("[blue]ℹ[/blue] Stats-only mode, skipping image display")


if __name__ == '__main__':
    main()
