#!/usr/bin/env python3
"""
Test script for thermal imaging human detection capabilities.

Tests the new thermal detection methods in the AgriculturalHumanDetector.
"""

import os
import sys
import cv2
import numpy as np
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agri_detector import AgriculturalHumanDetector

def create_test_thermal_frame(width=640, height=480):
    """
    Create a synthetic thermal frame for testing.

    Args:
        width: Frame width
        height: Frame height

    Returns:
        Synthetic thermal frame (single channel)
    """
    # Create base temperature field
    thermal_frame = np.random.normal(100, 20, (height, width)).astype(np.uint8)

    # Add some human-like thermal signatures
    # Human 1: Warm signature in center
    center_x, center_y = width // 2, height // 2
    human_temp = 140  # Human body temperature equivalent
    cv2.circle(thermal_frame, (center_x, center_y), 30, human_temp, -1)

    # Human 2: Smaller, cooler signature (farther away)
    far_x, far_y = width // 4, height // 4
    cv2.circle(thermal_frame, (far_x, far_y), 15, 120, -1)

    # Add some noise to simulate real thermal camera
    noise = np.random.normal(0, 5, thermal_frame.shape).astype(np.uint8)
    thermal_frame = cv2.add(thermal_frame, noise)

    # Ensure values are in valid range
    thermal_frame = np.clip(thermal_frame, 0, 255)

    return thermal_frame

def create_test_visible_frame(width=640, height=480):
    """
    Create a synthetic visible frame for testing.

    Args:
        width: Frame width
        height: Frame height

    Returns:
        Synthetic visible frame (BGR)
    """
    # Create a simple agricultural scene
    frame = np.ones((height, width, 3), dtype=np.uint8) * 100  # Gray background

    # Add some green field-like areas
    cv2.rectangle(frame, (0, height//2, width, height//2), (50, 100, 50), -1)

    # Add some brown dirt areas
    cv2.rectangle(frame, (width//3, height//3, width//3, height//3), (60, 40, 20), -1)

    return frame

def test_thermal_detection():
    """Test thermal detection capabilities."""
    print("Testing Thermal Detection Capabilities")
    print("=" * 50)

    # Initialize detector
    detector = AgriculturalHumanDetector()
    print(f"Detector initialized with thermal_enabled: {detector.thermal_enabled}")

    # Create test frames
    thermal_frame = create_test_thermal_frame()
    visible_frame = create_test_visible_frame()

    print(f"Created test thermal frame: {thermal_frame.shape}")
    print(f"Created test visible frame: {visible_frame.shape}")

    # Test thermal detection
    print("\n1. Testing thermal detection...")
    thermal_detections = detector.detect_thermal_humans(thermal_frame)
    print(f"Thermal detections found: {len(thermal_detections)}")

    for i, (bbox, conf, metadata) in enumerate(thermal_detections):
        print(".3f"
              f"Type: {metadata.get('thermal_type', 'unknown')}")

    # Test thermal simulation from visible
    print("\n2. Testing thermal simulation from visible...")
    simulated_thermal = detector._simulate_thermal_from_visible(visible_frame)
    print(f"Simulated thermal shape: {simulated_thermal.shape}")
    print(f"Simulated thermal range: {simulated_thermal.min()}-{simulated_thermal.max()}")

    # Test thermal preprocessing
    print("\n3. Testing thermal preprocessing...")
    processed_thermal = detector._preprocess_thermal_image(thermal_frame)
    print(f"Processed thermal shape: {processed_thermal.shape}")
    print(f"Processed thermal range: {processed_thermal.min()}-{processed_thermal.max()}")

    # Test thermal segmentation
    print("\n4. Testing thermal segmentation...")
    thermal_masks = detector._segment_thermal_regions(processed_thermal)
    print(f"Thermal masks created: {list(thermal_masks.keys())}")

    for mask_name, mask in thermal_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  {mask_name}: {len(contours)} contours")

    # Test multi-spectral fusion
    print("\n5. Testing multi-spectral fusion...")
    visible_detections = detector.detect_ultra_far_humans(visible_frame)
    print(f"Visible detections: {len(visible_detections)}")

    fused_detections = detector._fuse_multi_spectral_detections(
        thermal_detections=thermal_detections,
        visible_detections=visible_detections,
        thermal_frame=processed_thermal,
        visible_frame=visible_frame
    )
    print(f"Fused detections: {len(fused_detections)}")

    for i, (bbox, conf, metadata) in enumerate(fused_detections):
        method = metadata.get('detection_method', 'unknown')
        fusion_status = metadata.get('fusion_status', 'fused')
        print(".3f"
              f"Method: {method}, Status: {fusion_status}")

    # Test thermal signature analysis
    print("\n6. Testing thermal signature analysis...")
    if thermal_detections:
        bbox, conf, metadata = thermal_detections[0]
        thermal_stats = detector._analyze_thermal_signature(processed_thermal, bbox, 'human_warm')
        print("Thermal signature analysis:")
        for key, value in thermal_stats.items():
            if isinstance(value, float):
                print(".3f")
            else:
                print(f"  {key}: {value}")

        # Test human signature validation
        is_human = detector._is_human_thermal_signature(thermal_stats)
        print(f"Is human thermal signature: {is_human}")

    print("\nThermal Detection Test Complete!")
    return True

def test_thermal_integration():
    """Test thermal detection integration with main detection pipeline."""
    print("\nTesting Thermal Integration with Main Pipeline")
    print("=" * 50)

    detector = AgriculturalHumanDetector()

    # Create test visible frame
    visible_frame = create_test_visible_frame()

    # Test main detection with thermal enabled
    print("Running main detection pipeline...")
    detections = detector.detect(visible_frame)

    print(f"Total detections from main pipeline: {len(detections)}")

    # Check for thermal detections
    thermal_count = 0
    for bbox, conf, metadata in detections:
        if isinstance(metadata, dict) and metadata.get('detection_method') == 'thermal':
            thermal_count += 1

    print(f"Thermal detections in main pipeline: {thermal_count}")

    return True

if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        # Run tests
        test_thermal_detection()
        test_thermal_integration()

        print("\nAll thermal detection tests passed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)