"""
Quick test of YOLO detector directly on HackHPI images
"""

import cv2
import os
from ultralytics import YOLO

def test_yolo_direct():
    """Test YOLO directly on HackHPI image"""

    # Load YOLO model directly
    model = YOLO('yolov8n.pt')  # Use nano model that's already downloaded

    # Load first HackHPI image
    dataset_root = r"C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\HackHPI2026_release"
    data_dir = os.path.join(dataset_root, "data")

    # Find first image
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                break
        else:
            continue
        break

    print(f"Testing YOLO directly on: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    print(f"Image shape: {image.shape}")

    # Run YOLO detection
    results = model(image, classes=[0], conf=0.1, verbose=False)  # Very low confidence, person class only

    total_detections = 0
    for result in results:
        boxes = result.boxes
        total_detections += len(boxes)
        print(f"Result has {len(boxes)} boxes")

        # Show first few detections
        for i, box in enumerate(boxes[:3]):
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            print(f"  Detection {i}: class={cls}, bbox={bbox}, conf={conf:.3f}")

    print(f"Total YOLO detections: {total_detections}")

if __name__ == "__main__":
    test_yolo_direct()