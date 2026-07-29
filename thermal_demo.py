#!/usr/bin/env python3
"""
Simple demo script for enhanced agricultural human detection with thermal capabilities.
"""

import cv2
import numpy as np
from agri_detector import AgriculturalHumanDetector

def create_demo_frame():
    """Create a demo frame with synthetic human-like features."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some human-like shapes
    # Body
    cv2.rectangle(frame, (200, 150), (250, 300), (100, 100, 100), -1)
    # Head
    cv2.circle(frame, (225, 170), 20, (150, 150, 150), -1)
    # Arms
    cv2.rectangle(frame, (180, 180), (200, 250), (120, 120, 120), -1)
    cv2.rectangle(frame, (250, 180), (270, 250), (120, 120, 120), -1)

    return frame

def main():
    print("Enhanced Agricultural Human Detection Demo")
    print("=" * 50)

    # Initialize detector with thermal enabled
    detector = AgriculturalHumanDetector()
    print(f"Detector initialized with thermal capabilities: {detector.thermal_enabled}")

    # Create demo frames
    visible_frame = create_demo_frame()
    thermal_frame = detector._simulate_thermal_from_visible(visible_frame)

    print(f"Demo visible frame shape: {visible_frame.shape}")
    print(f"Demo thermal frame shape: {thermal_frame.shape}")

    # Run detection
    print("\nRunning enhanced detection pipeline...")
    detections = detector.detect(visible_frame)

    print(f"Total detections found: {len(detections)}")

    # Analyze detections
    for i, (bbox, confidence, metadata) in enumerate(detections):
        method = metadata.get('detection_method', 'unknown')
        thermal_type = metadata.get('thermal_type', 'N/A')
        print(".3f"
              f"Method: {method}, Thermal: {thermal_type}")

    # Test thermal-only detection
    print("\nTesting thermal-only detection...")
    thermal_detections = detector.detect_thermal_humans(thermal_frame)
    print(f"Thermal detections: {len(thermal_detections)}")

    # Test ultra-far detection
    print("\nTesting ultra-far detection...")
    far_detections = detector.detect_ultra_far_humans(visible_frame)
    print(f"Ultra-far detections: {len(far_detections)}")

    print("\nDemo completed successfully!")
    print("Enhanced capabilities:")
    print("- Ultra-far distance detection (100-150m)")
    print("- Sub-pixel accuracy with multi-scale processing")
    print("- Thermal imaging integration for night/low visibility")
    print("- Multi-spectral fusion for improved accuracy")
    print("- Advanced preprocessing and signature analysis")

if __name__ == "__main__":
    main()