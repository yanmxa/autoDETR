#!/usr/bin/env python3
"""
Image Capture System for Sign Detection Dataset Collection
Captures images from webcam for training sign language recognition models
"""

import cv2
import uuid
import time
import os
from datetime import datetime
from typing import List

from detr.utils import (
    display_capture_banner,
    display_capture_session_info,
    display_capture_session_summary,
    create_capture_progress,
    print_message
)


class CaptureImages:
    """Image capture system for collecting training data"""

    def __init__(self, path: str, classes: List[str], camera_id: int = 0) -> None:
        """
        Initialize the image capture system

        Args:
            path: Directory path to save captured images
            classes: List of class names to capture
            camera_id: Camera device ID (default: 0)
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        # Display banner using centralized function
        display_capture_banner()

        # Verify camera connection
        if not self.cap.isOpened():
            print_message(f"[bold red]✗[/bold red] Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            print_message(f"[bold green]✓[/bold green] Camera {camera_id} connected successfully")

        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        print_message(f"[blue]ℹ[/blue] Output directory: {self.path}")

    def capture(self, class_name: str) -> bool:
        """
        Capture a single image frame

        Args:
            class_name: Name of the class being captured

        Returns:
            bool: True if capture successful, False otherwise
        """
        try:
            ret, frame = self.cap.read()
            raw_frame = frame.copy()

            if not ret:
                raise Exception("Failed to read from camera")

            # Add text overlay to display frame
            image = cv2.putText(
                frame,
                f'Capturing {class_name}',
                (0, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                3,
                (0, 255, 0),  # Green color
                2,
                cv2.LINE_AA
            )
            cv2.imshow('Image Capture', image)

            # Create class directory if it doesn't exist
            class_dir = os.path.join(self.path, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Generate unique filename
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, raw_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print_message("[yellow]⚠[/yellow] Quit key pressed - stopping capture")
                return False

            return True

        except Exception as e:
            print_message(f"[bold red]✗[/bold red] Error capturing {class_name}: {str(e)}")
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        """
        Run the image capture session

        Args:
            sleep_time: Delay between captures in seconds
            num_images: Number of images to capture per class
        """
        # Display session information using centralized function
        display_capture_session_info(self.classes, num_images, sleep_time)

        total_captured = 0

        for img_class in self.classes:
            print_message(
                f"\n[bold magenta]┌─ Starting capture for: {img_class} ({num_images} images)[/bold magenta]"
            )

            class_captured = 0

            # Create progress bar using centralized function
            with create_capture_progress(show_time=False) as progress:
                task = progress.add_task(f"Capturing {img_class}", total=num_images)

                for idx in range(num_images):
                    success = self.capture(img_class)

                    if success:
                        class_captured += 1
                        total_captured += 1
                        print_message(
                            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
                            f"[green]✓[/green] Captured [bold]{img_class}[/bold] image #{idx + 1}"
                        )
                    else:
                        print_message(
                            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
                            f"[red]✗[/red] Error capturing {img_class} image #{idx + 1}"
                        )

                    progress.update(task, advance=1)
                    time.sleep(sleep_time)

            # Show completion for this class
            print_message(
                f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim] "
                f"[bold green]ℹ Completed {img_class}: {class_captured}/{num_images} images captured[/bold green]"
            )
            print_message(f"[magenta]└─ {img_class} complete[/magenta]\n")

        # Display session completion using centralized function
        display_capture_session_summary(total_captured, len(self.classes))

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print_message("[blue]ℹ[/blue] Camera released and windows closed")


def main():
    """Main entry point"""
    # Define classes to capture
    classes = ['one', 'two', 'three']

    # Initialize capture system (camera_id=0 for built-in camera)
    cap = CaptureImages('./data/raw', classes, camera_id=0)

    # Run capture session
    cap.run(sleep_time=2, num_images=30)


if __name__ == '__main__':
    main()
