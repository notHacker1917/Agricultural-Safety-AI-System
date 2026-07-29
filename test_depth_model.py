"""
Test script to demonstrate improved camera-depth spatial model for harvester safety.
Shows how Y-position (camera distance = truck distance) now drives risk scoring.
"""

import numpy as np
import cv2
import logging
from harvester_safety import HarvesterSafetyEngine
from harvester_visualizer import HarvesterSafetyVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_frame(width=640, height=480):
    """Create a blank test frame."""
    return np.ones((height, width, 3), dtype=np.uint8) * 200

def test_depth_model():
    """
    Test spatial model: Y-position (camera depth) is primary risk factor.
    
    Scenario:
    - Truck/harvester positioned at center-bottom of frame
    - Humans positioned at various Y-depths
    - Top of frame = far from camera = LOW RISK
    - Bottom of frame = close to camera = HIGH RISK
    """
    
    frame_width, frame_height = 640, 480
    harvester_position = (0.5, 0.7)  # Center-bottom
    
    safety_engine = HarvesterSafetyEngine(
        harvester_width=2.5,
        harvester_length=10.0,
        harvester_speed_ms=2.0,
        critical_forward_distance=30,
        critical_side_distance=5,
        warning_forward_distance=50,
        warning_side_distance=15
    )
    
    visualizer = HarvesterSafetyVisualizer()
    
    print("\n" + "="*70)
    print("HARVESTER SAFETY - DEPTH-WEIGHTED SPATIAL MODEL TEST")
    print("="*70)
    print("\nTest Scenario:")
    print(f"  Frame size: {frame_width}x{frame_height}")
    print(f"  Truck/Harvester position: center-bottom (x={harvester_position[0]}, y={harvester_position[1]})")
    print(f"  Critical zone: 0-30m from camera")
    print(f"  Warning zone: 30-50m from camera")
    print("\nTest Cases: Humans positioned at various Y-depths")
    print("-" * 70)
    
    # Test cases: humans at different Y-positions (camera depths)
    test_cases = [
        {
            'name': 'Human Far from Camera (Top of frame)',
            'bbox': [300, 50, 340, 130],  # Y: 50-130 (very far)
            'expected_risk': 'SAFE'
        },
        {
            'name': 'Human in Warning Zone (Middle)',
            'bbox': [300, 200, 340, 280],  # Y: 200-280 (warning depth)
            'expected_risk': 'WARNING'
        },
        {
            'name': 'Human in Critical Zone (Close to Camera/Truck)',
            'bbox': [300, 350, 340, 430],  # Y: 350-430 (very close)
            'expected_risk': 'CRITICAL'
        },
        {
            'name': 'Human Far Laterally but Warning Zone',
            'bbox': [100, 200, 140, 280],  # Lateral offset but same depth
            'expected_risk': 'WARNING'
        },
        {
            'name': 'Human Far Laterally and Critical Depth',
            'bbox': [520, 350, 560, 430],  # Lateral offset + critical depth
            'expected_risk': 'SAFE'  # Outside critical lateral zone (±5m), so not in danger zone
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        frame = create_test_frame(frame_width, frame_height)
        
        # Compute risk for this detection
        result = safety_engine.compute_risk_level(
            test_case['bbox'], 
            frame.shape,
            harvester_position
        )
        
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  BBox: {test_case['bbox']}")
        print(f"  Risk Level: {result['risk_level']} (Expected: {test_case['expected_risk']})")
        print(f"  Risk Score: {result['risk_score']:.3f}")
        print(f"  Camera Distance: {result['distance_m']:.1f}m")
        print(f"  Lateral Distance: {result['lateral_distance_m']:.1f}m")
        print(f"  Time-to-Collision: {result['time_to_collision_s']:.1f}s")
        print(f"  Details: {result['details']}")
        
        # Verify expectation
        match = "✓ PASS" if result['risk_level'] == test_case['expected_risk'] else "✗ FAIL"
        print(f"  {match}")
    
    print("\n" + "="*70)
    print("SPATIAL MODEL VALIDATION")
    print("="*70)
    print("""
KEY INSIGHTS:
1. Y-Position (Camera Depth) is PRIMARY risk factor
   - Higher Y value (closer to bottom) = closer to camera = closer to truck = MORE DANGER
   - Lower Y value (closer to top) = farther from camera = farther from truck = LESS DANGER

2. Lateral Distance (X-Position) is SECONDARY factor
   - Risk scales with distance from truck centerline (±5m critical, ±15m warning)
   - Only applies WITHIN the depth zones

3. Risk Score Weighting:
   - 70% weight on camera depth/truck distance
   - 30% weight on lateral distance
   - This matches real-world harvester dynamics

4. Time-to-Collision:
   - Based on camera distance and truck speed
   - Closer humans = shorter TTC = more urgent warnings
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_depth_model()
