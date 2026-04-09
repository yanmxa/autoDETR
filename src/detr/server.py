"""
Live webcam detection server using trained DETR model.

Captures frames from webcam, runs inference, draws bounding boxes,
and streams the annotated video to a browser via Flask.

Usage:
    python -m detr.server
    # Then open http://localhost:5001 in browser
"""

import cv2
import torch
import numpy as np
import time
from flask import Flask, Response, render_template_string

from detr.model import DETR
from detr.config import get_model_config, get_evaluation_config

CLASS_NAMES = ['one', 'two', 'three']
COLORS = [(0, 255, 0), (255, 165, 0), (0, 165, 255)]  # BGR: green, orange, blue

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

app = Flask(__name__)

# Global state
model = None
device = None
confidence_threshold = 0.95


def load_model():
    """Load the trained DETR model."""
    global model, device

    model_config = get_model_config()
    eval_config = get_evaluation_config()
    device = torch.device(eval_config['device'])

    model = DETR(
        num_classes=model_config['num_classes'],
        hidden_dim=model_config['hidden_dim'],
        nheads=model_config['nheads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_queries=model_config['num_queries'],
        dropout=model_config['dropout'],
        verbose=False,
    )

    checkpoint_path = eval_config['checkpoint_path']
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} on {device}")


def preprocess_frame(frame, image_size=224):
    """Convert a BGR webcam frame to model input tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size))
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)


def detect(frame):
    """Run detection on a single frame. Returns list of (class_name, confidence, x1, y1, x2, y2)."""
    h, w = frame.shape[:2]
    input_tensor = preprocess_frame(frame)

    with torch.no_grad():
        output = model(input_tensor)

    # Get full probabilities including background (last class)
    all_probs = output['pred_logits'].softmax(-1)[0]  # [num_queries, num_classes+1]
    bg_probs = all_probs[:, -1]  # background probability
    probs = all_probs[:, :-1]  # non-background probabilities
    max_probs, max_classes = probs.max(-1)

    boxes = output['pred_boxes'][0]  # [num_queries, 4] in cxcywh normalized

    detections = []
    for i in range(len(max_probs)):
        # Only detect if foreground class beats both threshold AND background
        if max_probs[i] > confidence_threshold and max_probs[i] > bg_probs[i]:
            cls_id = max_classes[i].item()
            conf = max_probs[i].item()
            cx, cy, bw, bh = boxes[i].cpu().numpy()

            # Convert normalized cxcywh to pixel xyxy
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            detections.append((CLASS_NAMES[cls_id], conf, x1, y1, x2, y2))

    # Keep only the top detection per class (highest confidence)
    best_per_class = {}
    for det in detections:
        cls_name = det[0]
        if cls_name not in best_per_class or det[1] > best_per_class[cls_name][1]:
            best_per_class[cls_name] = det

    return list(best_per_class.values())


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for cls_name, conf, x1, y1, x2, y2 in detections:
        cls_idx = CLASS_NAMES.index(cls_name)
        color = COLORS[cls_idx % len(COLORS)]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f'{cls_name}: {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def generate_frames():
    """Generator that yields MJPEG frames from webcam with detections."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam opened, streaming...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            detections = detect(frame)
            dt = (time.time() - t0) * 1000

            frame = draw_detections(frame, detections)

            # Draw FPS / inference time
            fps_label = f'{dt:.0f}ms | {len(detections)} det'
            cv2.putText(frame, fps_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>autoDETR Live Detection</title>
    <style>
        body {
            background: #1a1a2e; color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0; display: flex; flex-direction: column;
            align-items: center; justify-content: center; min-height: 100vh;
        }
        h1 { margin: 20px 0 10px; font-size: 1.5em; }
        .info { color: #888; font-size: 0.9em; margin-bottom: 15px; }
        img {
            border: 2px solid #333; border-radius: 8px;
            max-width: 90vw; max-height: 80vh;
        }
    </style>
</head>
<body>
    <h1>autoDETR Live Detection</h1>
    <div class="info">Classes: one, two, three | Threshold: {{ threshold }}</div>
    <img src="/video_feed" alt="Live Detection Feed">
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE, threshold=confidence_threshold)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    load_model()
    print("\nOpen http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    main()
