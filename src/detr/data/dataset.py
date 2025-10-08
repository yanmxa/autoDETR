"""
Dataset class for DETR object detection
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class DETRDataset(Dataset):
    """
    Dataset for DETR object detection.

    Expected directory structure:
        data_dir/
            ├── images/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── labels/
                ├── image1.txt
                ├── image2.txt
                └── ...

    Label format (YOLO format):
        Each line: <class_id> <x_center> <y_center> <width> <height>
        All coordinates are normalized to [0, 1]
    """

    def __init__(self, data_dir: str, train: bool = True, image_size: int = 224):
        """
        Args:
            data_dir: Root directory containing 'images' and 'labels' subdirectories
            train: Whether this is training dataset (affects augmentation)
            image_size: Target image size for training
        """
        super().__init__()
        self.data_dir = data_dir
        self.train = train
        self.image_size = image_size
        self.console = Console()

        # Paths
        self.images_path = os.path.join(data_dir, 'images')
        self.labels_path = os.path.join(data_dir, 'labels')

        # Get all label files
        label_files = os.listdir(self.labels_path)
        self.labels = [f for f in label_files if f.endswith('.txt')]

        # Display dataset info with rich
        self._display_dataset_info()

    def _display_dataset_info(self):
        """Display dataset initialization information using rich."""
        # Create dataset statistics table
        table = Table(title="📊 DETR Dataset Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow")

        mode_emoji = "🏋️" if self.train else "🧪"
        mode_text = f"{mode_emoji} {'Training' if self.train else 'Validation/Test'}"

        table.add_row("Data Path", self.data_dir)
        table.add_row("Mode", mode_text)
        table.add_row("Total Samples", str(len(self.labels)))
        table.add_row("Image Size", f"{self.image_size}×{self.image_size}")
        table.add_row("Images Path", self.images_path)
        table.add_row("Labels Path", self.labels_path)

        self.console.print(table)

        # Display augmentation info
        if self.train:
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
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()  # Empty line for spacing

    def _get_transform(self):
        """Get albumentations transform pipeline."""
        if self.train:
            # Training augmentations
            transform = A.Compose([
                A.Resize(500, 500),
                A.RandomCrop(width=self.image_size, height=self.image_size, p=0.33),
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # Validation/test transforms (no augmentation)
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        return transform

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        """
        Apply transform with retry mechanism to handle edge cases.

        Args:
            image: Input image (numpy array)
            bboxes: Bounding boxes in YOLO format
            labels: Class labels
            max_attempts: Maximum number of retry attempts

        Returns:
            Dictionary with transformed image, bboxes, and labels
        """
        transform = self._get_transform()

        for _ in range(max_attempts):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
                # Check if we still have bboxes after transformation
                if len(transformed['bboxes']) > 0:
                    return transformed
            except Exception:
                # Continue trying with different random seeds
                continue

        # Fallback: return original with minimal transform
        fallback_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        return fallback_transform(image=image, bboxes=bboxes, class_labels=labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, target_dict)
            where target_dict contains 'labels' and 'boxes'
        """
        # Get label file path
        label_file = self.labels[idx]
        label_path = os.path.join(self.labels_path, label_file)

        # Get corresponding image path
        image_name = label_file.split('.')[0]
        image_path = os.path.join(self.images_path, f'{image_name}.jpg')

        # Load image
        img = Image.open(image_path).convert('RGB')

        # Load annotations
        with open(label_path, 'r') as f:
            annotations = f.readlines()

        class_labels = []
        bounding_boxes = []

        for annotation in annotations:
            # Parse annotation line
            parts = annotation.strip().split()
            if len(parts) >= 5:
                class_labels.append(int(parts[0]))
                bounding_boxes.append([float(x) for x in parts[1:5]])

        # Convert to numpy arrays
        class_labels = np.array(class_labels, dtype=np.int64)
        bounding_boxes = np.array(bounding_boxes, dtype=np.float32)

        # Apply transformations
        augmented = self.safe_transform(
            image=np.array(img),
            bboxes=bounding_boxes,
            labels=class_labels
        )

        # Extract transformed data
        image_tensor = augmented['image']
        transformed_boxes = np.array(augmented['bboxes'], dtype=np.float32)
        transformed_labels = np.array(augmented['class_labels'], dtype=np.int64)

        # Convert to tensors
        labels_tensor = torch.tensor(transformed_labels, dtype=torch.long)
        boxes_tensor = torch.tensor(transformed_boxes, dtype=torch.float32)

        # Return in DETR format
        target = {
            'labels': labels_tensor,
            'boxes': boxes_tensor
        }

        return image_tensor, target


def collate_fn(batch):
    """
    Custom collate function for batching samples with varying number of objects.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (images_tensor, targets_list)
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)

    return images, targets


if __name__ == '__main__':
    """Test the dataset."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Create dataset
    dataset = DETRDataset('data/train', train=True)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, drop_last=True)

    # Get a batch
    images, targets = next(iter(dataloader))

    print(f"Batch shape: {images.shape}")
    print(f"Number of targets: {len(targets)}")

    # Visualize first 4 images
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    for idx, (img, target, ax) in enumerate(zip(images, targets, axes)):
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # Display image
        ax.imshow(img_denorm.permute(1, 2, 0).numpy())
        ax.set_title(f'Sample {idx + 1}')
        ax.axis('off')

        # Draw bounding boxes
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        img_h, img_w = img_denorm.shape[1], img_denorm.shape[2]

        for box, label in zip(boxes, labels):
            # Convert from YOLO format (cx, cy, w, h) to corner format
            cx, cy, w, h = box
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            box_w = w * img_w
            box_h = h * img_h

            # Draw rectangle
            rect = Rectangle(
                (x1, y1), box_w, box_h,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(
                x1, y1 - 5,
                f'Class {label}',
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=10,
                color='black'
            )

    plt.tight_layout()
    plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'dataset_sample.png'")
    plt.show()
