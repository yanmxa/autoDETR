"""
DETR Real-time Detection Tool

Real-time object detection using webcam feed with the trained DETR model.
Displays only the highest confidence detection per frame.

Usage:
    python -m detr.tools.realtime_detect

    Or as a module:
    from detr.tools import realtime_detect
    realtime_detect()

Key Features:
    - Shows ONLY the highest confidence detection per frame
    - Displays FPS and inference time
    - Auto-detects camera resolution
    - Press 'q' to quit

Configuration (edit get_realtime_config() below):
    - camera_id: 0 (default webcam) or video file path
    - confidence_threshold: 0.1-0.9 (lower = more detections)
    - show_fps: True/False
    - show_inference_time: True/False
    - show_detection_info: True/False (periodic debug output)
    - box_thickness: 1-10
    - font_scale: 0.5-2.0

Notes:
    - Uses actual camera frame size for bbox scaling (not hardcoded)
    - Model input is always 224x224 (from training config)
    - Lower confidence_threshold (e.g., 0.1-0.3) for initial testing
    - Increase threshold (0.5-0.8) for production use
"""

import cv2
import torch
import time
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from detr.model import DETR
from detr.config import get_evaluation_config
from detr.utils import print_message, print_empty_line


def get_realtime_config():
    """
    Get real-time detection configuration.

    Returns:
        dict: Real-time detection configuration

    Adjustable Parameters:
        camera_id: int or str
            - 0 = default webcam
            - 1, 2, ... = other cameras
            - 'video.mp4' = video file path

        confidence_threshold: float (0.0-1.0)
            - Lower (0.1-0.3): More sensitive, may show false positives
            - Medium (0.3-0.5): Balanced
            - Higher (0.5-0.8): Only high-confidence detections

        show_fps: bool - Display FPS counter
        show_inference_time: bool - Display inference time in ms
        show_detection_info: bool - Print debug info every 30 frames

        Visual Settings:
            box_thickness: int (1-10) - Bounding box line width
            font_scale: float (0.5-2.0) - Text size
            font_thickness: int (1-4) - Text line width
    """
    # Start with evaluation config
    config = get_evaluation_config()

    # Update for real-time detection
    config.update({
        # Camera settings
        'camera_id': 0,  # 0=webcam, or use video file path

        # Detection settings
        'confidence_threshold': 0.0,  # Start low, increase for better quality

        # Performance tracking
        'fps_update_interval': 30,  # Update FPS display every N frames

        # Display options
        'show_fps': True,
        'show_inference_time': True,
        'show_detection_info': True,  # Periodic debug output to console

        # Visual appearance
        'box_thickness': 3,
        'font_scale': 1.0,
        'font_thickness': 2,
        'label_height': 40,  # Internal use
        'label_width': 300,  # Internal use
    })

    return config


def rescale_bboxes(boxes, img_size):
    """
    Rescale bounding boxes from normalized [0, 1] to pixel coordinates.

    Args:
        boxes: Tensor of shape [N, 4] in format [cx, cy, w, h] normalized
        img_size: Tuple (width, height) of target image size

    Returns:
        Tensor of shape [N, 4] in format [xmin, ymin, xmax, ymax] in pixels
    """
    w, h = img_size

    # Convert from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # xmin = cx - w/2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # ymin = cy - h/2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # xmax = cx + w/2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # ymax = cy + h/2

    # Scale to image size
    boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], dtype=boxes.dtype)

    return boxes_xyxy


def get_transforms(image_size=224):
    """
    Get image transformations for real-time inference.

    Args:
        image_size: Target image size for model input

    Returns:
        albumentations.Compose: Transformation pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_class_names():
    """
    Get class names for detection.

    Returns:
        list: List of class names
    """
    # TODO: Load from config file if available
    return ['one', 'two', 'three']


def get_class_colors():
    """
    Get colors for each class (BGR format for OpenCV).

    Returns:
        list: List of BGR color tuples
    """
    return [
        (0, 255, 0),    # Green for 'one'
        (255, 0, 0),    # Blue for 'two'
        (0, 0, 255),    # Red for 'three'
    ]


def draw_detections(frame, classes, probas, bboxes, class_names, colors, config):
    """
    Draw bounding boxes and labels on frame.

    Args:
        frame: OpenCV image (BGR)
        classes: Tensor of class indices
        probas: Tensor of confidence scores
        bboxes: Tensor of bounding boxes [xmin, ymin, xmax, ymax]
        class_names: List of class names
        colors: List of BGR colors
        config: Configuration dictionary

    Returns:
        frame: Annotated frame
        detections: List of detection dictionaries
    """
    detections = []

    for bclass, bprob, bbox in zip(classes, probas, bboxes):
        bclass_idx = bclass.detach().cpu().numpy()
        bprob_val = bprob.detach().cpu().numpy()
        x1, y1, x2, y2 = bbox.detach().cpu().numpy()

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Clamp bounding box to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Store detection info
        detections.append({
            'class': class_names[bclass_idx],
            'confidence': float(bprob_val),
            'bbox': [x1, y1, x2, y2]
        })

        # Get color for this class
        color = colors[bclass_idx]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config['box_thickness'])

        # Prepare label text
        label_text = f"{class_names[bclass_idx]}: {bprob_val:.3f}"

        # Get text size for better label placement
        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            config['font_scale'],
            config['font_thickness']
        )

        # Draw label background
        label_y1 = max(y1 - text_h - 10, 0)
        label_x2 = min(x1 + text_w + 10, w)

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (label_x2, y1),
            color,
            -1  # Filled rectangle
        )

        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 5),  # Slight padding
            cv2.FONT_HERSHEY_SIMPLEX,
            config['font_scale'],
            (255, 255, 255),  # White text
            config['font_thickness'],
            cv2.LINE_AA
        )

    return frame, detections


def run_realtime_detection(config):
    """
    Main real-time detection loop.

    Args:
        config: Configuration dictionary
    """
    print_message("\n[bold cyan]DETR Real-time Detection[/bold cyan]\n")

    # Initialize model
    print_message("[bold]Initializing model...[/bold]")
    model = DETR(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        nheads=config['nheads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_queries=config['num_queries'],
        dropout=config['dropout'],
        verbose=False
    )

    # Load checkpoint
    print_message(f"[bold]Loading checkpoint:[/bold] [cyan]{config['checkpoint_path']}[/cyan]")
    model.load_pretrained(config['checkpoint_path'])

    # Move to device and set to eval mode
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    print_message(f"[green]✓[/green] Model ready on device: [yellow]{device}[/yellow]")
    print_empty_line()

    # Get transforms
    transforms = get_transforms(config.get('image_size', 224))

    # Get class names and colors
    class_names = get_class_names()
    colors = get_class_colors()

    # Initialize camera
    print_message(f"[bold]Starting camera capture (ID: {config['camera_id']})...[/bold]")
    cap = cv2.VideoCapture(config['camera_id'])

    if not cap.isOpened():
        print_message("[red]✗[/red] Failed to open camera")
        return

    # Get actual camera frame size
    ret, test_frame = cap.read()
    if not ret:
        print_message("[red]✗[/red] Failed to read test frame from camera")
        return

    frame_height, frame_width = test_frame.shape[:2]
    print_message(f"[green]✓[/green] Camera opened successfully")
    print_message(f"[blue]ℹ[/blue] Camera resolution: [yellow]{frame_width}x{frame_height}[/yellow]")
    print_message(f"[blue]ℹ[/blue] Confidence threshold: [yellow]{config['confidence_threshold']:.2f}[/yellow]")
    print_message("[yellow]Press 'q' to quit[/yellow]")
    print_empty_line()

    # Performance tracking
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    max_detections_seen = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print_message("[red]✗[/red] Failed to read frame from camera")
                break

            # Get actual frame dimensions
            h, w = frame.shape[:2]

            # Run inference
            inference_start = time.time()

            # Transform image
            transformed = transforms(image=frame)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            # Model inference
            with torch.no_grad():
                result = model(image_tensor)

            inference_time = (time.time() - inference_start) * 1000  # Convert to ms

            # Filter predictions - Get only the highest confidence detection
            probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]  # Exclude background
            max_probs, max_classes = probabilities.max(-1)

            # Find the single highest confidence detection across all queries
            highest_conf_value = max_probs.max().item()

            # Debug: Print max confidence every 30 frames
            if config.get('show_detection_info') and frame_count % 30 == 0:
                print_message(f"[dim]Frame {frame_count}: Max confidence: {highest_conf_value:.3f}[/dim]")

            # Only show detection if it exceeds threshold
            if highest_conf_value > config['confidence_threshold']:
                # Get the index of highest confidence detection
                batch_idx, query_idx = torch.where(max_probs == max_probs.max())

                # Take only the first one if there are ties
                batch_idx = batch_idx[0].item()
                query_idx = query_idx[0].item()

                # Get the single bbox - shape is [1, 4] after indexing
                single_bbox = result['pred_boxes'][batch_idx, query_idx].unsqueeze(0)

                # Use ACTUAL frame size for rescaling
                bbox = rescale_bboxes(single_bbox, (w, h))

                # Get class and probability
                cls = max_classes[batch_idx, query_idx].unsqueeze(0)
                prob = max_probs[batch_idx, query_idx].unsqueeze(0)

                # Draw detection
                frame, detections = draw_detections(
                    frame, cls, prob, bbox,
                    class_names, colors, config
                )

                max_detections_seen = max(max_detections_seen, len(detections))
            else:
                detections = []

            # Update FPS counter
            frame_count += 1
            if frame_count % config['fps_update_interval'] == 0:
                elapsed_time = time.time() - fps_start_time
                fps = config['fps_update_interval'] / elapsed_time
                fps_start_time = time.time()

            # Draw FPS and inference time on frame
            info_y = 50
            if config['show_fps']:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    frame, fps_text, (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA
                )
                info_y += 50

            if config['show_inference_time']:
                inf_text = f"Inference: {inference_time:.1f}ms"
                cv2.putText(
                    frame, inf_text, (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA
                )
                info_y += 50

            # Draw detection count
            det_text = f"Detections: {len(detections)}"
            cv2.putText(
                frame, det_text, (20, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA
            )

            # Display frame
            cv2.imshow('DETR Real-time Detection', frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print_message("\n[yellow]⚠[/yellow] Stopping real-time detection...")
                break

    except KeyboardInterrupt:
        print_message("\n[yellow]⚠[/yellow] Interrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print session summary
        print_message("\n[bold cyan]Session Summary:[/bold cyan]")
        print_message(f"  Total frames processed: [yellow]{frame_count}[/yellow]")
        print_message(f"  Max detections in single frame: [yellow]{max_detections_seen}[/yellow]")
        print_message("\n[bold green]✓ Real-time detection stopped[/bold green]\n")


def main():
    """Main entry point."""
    config = get_realtime_config()
    run_realtime_detection(config)


if __name__ == '__main__':
    main()
