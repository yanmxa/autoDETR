"""
Inference script for real-time sign detection
"""

import argparse
import torch
import cv2
from pathlib import Path
from rich.console import Console

from sign_detection.models import SignDetectionModel
from sign_detection.data.transforms import get_val_transforms
from PIL import Image


class SignDetector:
    """Real-time sign detector"""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.console = Console()

        # Load checkpoint
        self.console.print(f"[blue]Loading model from {checkpoint_path}[/blue]")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.classes = checkpoint.get('classes', [])
        num_classes = len(self.classes)

        # Create model
        self.model = SignDetectionModel(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Transform
        self.transform = get_val_transforms()

        self.console.print(f"[green]Model loaded successfully[/green]")
        self.console.print(f"[blue]Classes: {', '.join(self.classes)}[/blue]")

    @torch.no_grad()
    def predict(self, image):
        """Predict sign from image"""
        # Preprocess
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, cv2.Mat):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        outputs = self.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

        return {
            'class': self.classes[predicted_idx],
            'confidence': confidence,
            'probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probabilities)}
        }

    def run_webcam(self, camera_id: int = 0):
        """Run inference on webcam feed"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            self.console.print(f"[red]Failed to open camera {camera_id}[/red]")
            return

        self.console.print("[green]Starting webcam inference. Press 'q' to quit.[/green]")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            result = self.predict(frame)

            # Display result
            text = f"{result['class']}: {result['confidence']:.2f}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow('Sign Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Sign detection inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file')
    parser.add_argument('--webcam', action='store_true',
                        help='Run on webcam')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    console = Console()
    detector = SignDetector(args.checkpoint, args.device)

    if args.webcam:
        detector.run_webcam(args.camera_id)
    elif args.image:
        result = detector.predict(args.image)
        console.print(f"\n[bold]Prediction:[/bold] {result['class']}")
        console.print(f"[bold]Confidence:[/bold] {result['confidence']:.4f}\n")

        # Show all probabilities
        console.print("[bold]All probabilities:[/bold]")
        for class_name, prob in result['probabilities'].items():
            console.print(f"  {class_name}: {prob:.4f}")
    else:
        console.print("[red]Please specify either --image or --webcam[/red]")


if __name__ == '__main__':
    main()
