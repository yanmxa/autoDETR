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

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich import box


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
        self.console = Console()
        self.cap = cv2.VideoCapture(camera_id)
        self.path = path
        self.classes = classes

        # Print banner
        self._print_banner()

        # Verify camera connection
        if not self.cap.isOpened():
            self.console.print(f"[bold red]✗[/bold red] Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            self.console.print(f"[bold green]✓[/bold green] Camera {camera_id} connected successfully")

        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        self.console.print(f"[blue]ℹ[/blue] Output directory: {self.path}")

    def _print_banner(self):
        """Print beautiful banner at startup"""
        banner = Panel.fit(
            "[bold cyan]IMAGE CAPTURE SYSTEM v1.0[/bold cyan]\n"
            "[dim]Sign Language Dataset Collection[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        )
        self.console.print(banner)

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
                self.console.print("[yellow]⚠[/yellow] Quit key pressed - stopping capture")
                return False

            return True

        except Exception as e:
            self.console.print(f"[bold red]✗[/bold red] Error capturing {class_name}: {str(e)}")
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        """
        Run the image capture session

        Args:
            sleep_time: Delay between captures in seconds
            num_images: Number of images to capture per class
        """
        # Display session information
        self._print_session_info(num_images, sleep_time)

        total_captured = 0
        start_time = time.time()

        for class_idx, img_class in enumerate(self.classes):
            self.console.print(
                f"\n[bold magenta]┌─ Starting capture for: {img_class} ({num_images} images)[/bold magenta]"
            )

            class_captured = 0

            # Create progress bar for this class
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Capturing {img_class}", total=num_images)

                for idx in range(num_images):
                    success = self.capture(img_class)

                    if success:
                        class_captured += 1
                        total_captured += 1
                        self.console.print(
                            f"[dim]{datetime.now().strftime('%m/%d/%y %H:%M:%S')}[/dim] "
                            f"[green]✓[/green] Captured [bold]{img_class}[/bold] image #{idx + 1}"
                        )
                    else:
                        self.console.print(
                            f"[dim]{datetime.now().strftime('%m/%d/%y %H:%M:%S')}[/dim] "
                            f"[red]✗[/red] Error capturing {img_class} image #{idx + 1}"
                        )

                    progress.update(task, advance=1)
                    time.sleep(sleep_time)

            # Show completion for this class
            self.console.print(
                f"[dim]{datetime.now().strftime('%m/%d/%y %H:%M:%S')}[/dim] "
                f"[bold green]ℹ Completed {img_class}: {class_captured}/{num_images} images captured[/bold green]"
            )
            self.console.print(f"[magenta]└─ {img_class} complete[/magenta]\n")

        # Show session completion
        self._print_session_summary(total_captured, len(self.classes))

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        self.console.print("[blue]ℹ[/blue] Camera released and windows closed")

    def _print_session_info(self, num_images: int, sleep_time: int):
        """Print capture session start information"""
        table = Table(title="Capture Session Configuration", box=box.ROUNDED, border_style="cyan")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold")

        table.add_row("Classes", ", ".join(self.classes))
        table.add_row("Images per class", str(num_images))
        table.add_row("Interval", f"{sleep_time}s")
        table.add_row("Total images", str(num_images * len(self.classes)))

        self.console.print()
        self.console.print(table)
        self.console.print()

    def _print_session_summary(self, total_captured: int, num_classes: int):
        """Print session completion summary"""
        summary = Panel(
            f"[bold green]✓[/bold green] Total images captured: [bold]{total_captured}[/bold]\n"
            f"[bold green]✓[/bold green] Classes processed: [bold]{num_classes}[/bold]",
            title="[bold]Session Complete[/bold]",
            border_style="green",
            box=box.DOUBLE
        )
        self.console.print()
        self.console.print(summary)


def main():
    """Main entry point"""
    # Define classes to capture
    classes = ['one', 'two', 'three']

    # Initialize capture system (camera_id=0 for built-in camera)
    cap = CaptureImages('./data/raw/test1', classes, camera_id=0)

    # Run capture session
    cap.run(sleep_time=2, num_images=30)


if __name__ == '__main__':
    main()
