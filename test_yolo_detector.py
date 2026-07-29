"""
Quick test of YOLO detector on HackHPI images
"""

import cv2
import os
from agri_detector import AgriculturalHumanDetector
from pathlib import Path

# Test on first few images
dataset_root = r"C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\HackHPI2026_release"
data_dir = os.path.join(dataset_root, "data")

def test_detector():
    # Initialize detector with YOLO only (disable other methods)
    detector = AgriculturalHumanDetector(
        model_path='yolov8n.pt',
        conf=0.1,  # Very low confidence
        use_preprocessing=False,  # Disable preprocessing
        enable_far_detection=True
    )
    
    # Temporarily disable other detection methods to isolate YOLO
    detector.thermal_enabled = False
    detector.super_resolution_enabled = False
    detector.frequency_domain_enhancement = False
    detector.adversarial_noise_reduction = False
    
    # Try to disable ultra-far methods by setting flags
    detector.ultra_far_scales = []  # Disable ultra-far scaling

    # Find first image
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                print(f"Testing detector on: {image_path}")

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print("Failed to load image")
                    continue

                print(f"Image shape: {image.shape}")

                # Run detection
                detections = detector.detect(image)
                print(f"Detections found: {len(detections)}")

                for i, det in enumerate(detections[:5]):  # Show first 5
                    if len(det) == 4:
                        bbox, method, conf, rel_size = det
                        print(f"  Detection {i}: bbox={bbox}, method={method}, conf={conf:.3f}, rel_size={rel_size:.4f}")
                    elif len(det) == 3:
                        bbox, conf, extra = det
                        print(f"  Detection {i}: bbox={bbox}, conf={conf:.3f}, extra={extra}")
                    else:
                        print(f"  Detection {i}: {det}")

                # Only test first image
                return

if __name__ == "__main__":
    test_detector()