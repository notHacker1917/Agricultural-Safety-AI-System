#!/usr/bin/env python3
"""
Quick test for ultra-far detection capabilities.
"""

import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agri_detector import AgriculturalHumanDetector

def create_test_frame():
    """Create a test frame with tiny human-like objects."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [100, 150, 200]  # Light background

    # Add very small human-like shapes (simulating far distance)
    tiny_humans = [
        (50, 100, 65, 125),    # 15x25 pixels - very far
        (200, 80, 230, 120),   # 30x40 pixels - far
        (400, 150, 440, 200),  # 40x50 pixels - medium
    ]

    for x1, y1, x2, y2 in tiny_humans:
        # Draw tiny human shape
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 120, 160), -1)  # Body
        head_size = min(x2-x1, y2-y1) // 4
        if head_size > 2:
            cv2.circle(frame, ((x1+x2)//2, y1 + head_size), head_size, (180, 140, 100), -1)

    return frame

def main():
    print("🚀 Quick Ultra-Far Detection Test")
    print("=" * 40)

    # Initialize detector
    detector = AgriculturalHumanDetector(conf=0.05, enable_far_detection=True)

    # Create test frame
    test_frame = create_test_frame()
    print(f"Created test frame: {test_frame.shape}")
    print("Added 3 tiny human-like objects (15-50 pixels)")

    # Test ultra-far detection
    print("\n🔍 Testing Ultra-Far Detection...")
    detections = detector.detect_ultra_far_humans(test_frame)

    print(f"Ultra-far detections found: {len(detections)}")

    for i, (bbox, conf, metadata) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        print(".3f"
              f"Size: {width:.0f}x{height:.0f}px")

    # Test main detection pipeline
    print("\n🔄 Testing Main Detection Pipeline...")
    all_detections = detector.detect(test_frame)

    print(f"Total detections from main pipeline: {len(all_detections)}")

    for i, (bbox, conf, metadata) in enumerate(all_detections):
        method = metadata.get('detection_method', 'unknown')
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        print(".3f"
              f"Method: {method}")

    print("\n✅ Ultra-Far Detection Test Complete!")

if __name__ == "__main__":
    main()