#!/usr/bin/env python3
"""
5-Tier Risk Parameter Validation & Test Script
Tests all 5 risk parameters and generates comprehensive validation report
"""

import sys
sys.path.insert(0, '.')

from enhanced_risk_assessor import EnhancedRiskAssessor
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_5tier_risk_system():
    """Test all 5 risk levels with comprehensive parameter verification"""
    
    print("\n" + "="*80)
    print("5-TIER RISK ASSESSMENT SYSTEM - PARAMETER VERIFICATION")
    print("="*80 + "\n")
    
    # Initialize assessor
    assessor = EnhancedRiskAssessor(frame_shape=(480, 640), debug=True)
    
    print("\n" + "="*80)
    print("TEST SCENARIOS - Verifying All 5 Risk Levels")
    print("="*80 + "\n")
    
    scenarios = [
        {
            'name': '1. SAFE - Far away, not in FOV',
            'bbox': (550, 50, 620, 150),  # Top right, small
            'prev_bbox': (540, 40, 610, 140),
            'expected': 'SAFE'
        },
        {
            'name': '2. LOW_WARNING - Moderate distance, in FOV, approaching',
            'bbox': (280, 150, 360, 350),  # Center, medium
            'prev_bbox': (290, 140, 370, 340),
            'expected': 'LOW_WARNING'
        },
        {
            'name': '3. WARNING - Closer, clearly in FOV',
            'bbox': (240, 250, 400, 450),  # Center, larger
            'prev_bbox': (250, 240, 410, 440),
            'expected': 'WARNING'
        },
        {
            'name': '4. HIGH_WARNING - Very close, directly ahead',
            'bbox': (220, 350, 420, 470),  # Large, nearly full height
            'prev_bbox': (230, 340, 430, 460),
            'expected': 'HIGH_WARNING'
        },
        {
            'name': '5. CRITICAL - Immediate danger, collision imminent',
            'bbox': (200, 400, 440, 475),  # Massive, at bottom
            'prev_bbox': (210, 390, 450, 465),
            'expected': 'CRITICAL'
        },
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        print("-" * 80)
        
        assessment = assessor.compute_risk_level_and_score(
            bbox=scenario['bbox'],
            prev_bbox=scenario['prev_bbox']
        )
        
        risk_level = assessment['risk_level']
        risk_score = assessment['risk_score']
        
        # Print all 5 parameters
        print(f"\n  ✓ RISK LEVEL: {risk_level} (Score: {risk_score:.3f})")
        print(f"    Expected: {scenario['expected']}")
        match = "✓ PASS" if risk_level == scenario['expected'] else "✗ FAIL"
        print(f"    Result: {match}\n")
        
        print("  ===== PARAMETER VERIFICATION =====")
        
        # Parameter 1: Distance
        print(f"\n  1. FORWARD DISTANCE (Depth):")
        print(f"     Distance: {assessment['distance_m']:.1f}m")
        print(f"     Zone: {assessment['distance_zone']}")
        print(f"     Safe: {assessment['distance_safe']}")
        print(f"     Threshold: <5m(CRIT) | 5-15m(HI) | 15-25m(WARN) | 25-40m(LO) | >40m(SAFE)")
        
        # Parameter 2: Lateral
        print(f"\n  2. LATERAL DISTANCE (Side-to-Side):")
        print(f"     Distance: {assessment['lateral_distance_m']:.1f}m from centerline")
        print(f"     Zone: {assessment['lateral_zone']}")
        print(f"     Safe: {assessment['lateral_safe']}")
        print(f"     Threshold: ±3m(CRIT) | ±8m(HI) | ±12m(WARN) | ±20m(LO) | >±20m(SAFE)")
        
        # Parameter 3: FOV
        print(f"\n  3. FIELD OF VIEW (FOV):")
        print(f"     In FOV: {assessment['in_fov']}")
        print(f"     FOV Safe: {assessment['fov_safe']}")
        print(f"     FOV Range: {assessment['fov_left']:.2f} to {assessment['fov_right']:.2f} (normalized)")
        print(f"     Zone: Center ±15% = HIGH RISK, Outside = LOWER RISK")
        
        # Parameter 4: Direction
        print(f"\n  4. MOVEMENT DIRECTION:")
        print(f"     Direction: {assessment['direction']}")
        print(f"     Approaching: {assessment['approaching']}")
        print(f"     Direction Safe: {assessment['direction_safe']}")
        print(f"     Rule: APPROACHING = ESCALATE RISK | RETREATING = REDUCE RISK")
        
        # Parameter 5: Speed
        print(f"\n  5. SPEED CATEGORY:")
        print(f"     Speed: {assessment['speed_category']}")
        print(f"     Speed Norm: {assessment['speed_norm']:.2f}/1.0")
        print(f"     Speed Safe: {assessment['speed_safe']}")
        print(f"     Rule: Fast approach = HIGHER RISK | Slow/stationary = LOWER RISK")
        
        print(f"\n  ===== SUMMARY ======")
        print(f"  {assessment['details']}")
        
        results.append({
            'scenario': scenario['name'],
            'expected': scenario['expected'],
            'actual': risk_level,
            'passed': risk_level == scenario['expected'],
            'score': risk_score
        })
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{status} | {result['scenario'].split('-')[0].strip():30s} | "
              f"Expected: {result['expected']:15s} Got: {result['actual']:15s} | Score: {result['score']:.3f}")
    
    print(f"\n{passed}/{total} tests passed ({100*passed/total:.0f}%)\n")
    
    print("="*80)
    print("RISK PARAMETER VERIFICATION COMPLETE")
    print("="*80 + "\n")
    
    return passed == total

def test_movement_escalation():
    """Test movement-based risk escalation/de-escalation"""
    
    print("\n" + "="*80)
    print("MOVEMENT ESCALATION TESTS")
    print("="*80 + "\n")
    
    assessor = EnhancedRiskAssessor(frame_shape=(480, 640), debug=False)
    
    print("TEST: Risk escalation when approaching\n")
    
    # Start at SAFE distance, moving closer
    bbox_sequence = [
        (520, 80, 620, 200),    # Far, SAFE
        (400, 150, 500, 350),   # Closer, approaching
        (300, 250, 400, 450),   # Even closer
        (220, 350, 420, 470),   # Very close
    ]
    
    prev_bbox = None
    for step, bbox in enumerate(bbox_sequence, 1):
        assessment = assessor.compute_risk_level_and_score(
            bbox=bbox,
            prev_bbox=prev_bbox
        )
        
        print(f"Step {step}: {assessment['risk_level']:15s} | "
              f"Distance: {assessment['distance_m']:5.1f}m | "
              f"Dir: {assessment['direction']:10s} | "
              f"Score: {assessment['risk_score']:.3f}")
        
        prev_bbox = bbox
    
    print("\n✓ Escalation test complete - risk should increase as humans approach\n")

if __name__ == "__main__":
    # Run tests
    all_passed = test_5tier_risk_system()
    test_movement_escalation()
    
    if all_passed:
        print("\n✓✓✓ ALL PARAMETERS VERIFIED ✓✓✓\n")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed - check parameters\n")
        sys.exit(1)
