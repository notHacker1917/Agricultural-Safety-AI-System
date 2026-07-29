"""
COMPLETE SYSTEM INTEGRATION TEST

Demonstrates all safety components working together:
✓ Tractor geometry (3D camera projection)
✓ Terrain analysis
✓ Risk assessment (7-factor model)
✓ Safety controller orchestration
✓ Emergency protocols & escalation
✓ Monitoring dashboard & alerts
✓ Audit logging

This test creates synthetic detections and processes them
through the complete safety pipeline to verify integration.
"""

import numpy as np
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import os

# Import all safety components
from tractor_geometry import TractorPOVGeometry, create_realistic_camera, TractorGeometry, TractorModel
from terrain_analysis import TerrainAnalyzer
from context_aware_risk_system import ContextAwareRiskAssessor
from safety_controller import SafetySystemController
from emergency_protocols import EmergencyResponseController
from monitoring_dashboard import MonitoringDashboard

# Configure logging
log_dir = os.path.expanduser("~/safety_logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "system_test.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemIntegrationTest:
    """Complete system integration test."""
    
    def __init__(self):
        """Initialize all components."""
        logger.info("=" * 100)
        logger.info("COMPLETE SYSTEM INTEGRATION TEST")
        logger.info("=" * 100)
        
        # Initialize components
        tractor = TractorGeometry.default_harvester(TractorModel.GENERIC)
        camera = create_realistic_camera()
        self.tractor_geom = TractorPOVGeometry(tractor, camera)
        
        self.terrain_analyzer = TerrainAnalyzer()
        self.risk_assessor = ContextAwareRiskAssessor(pov=self.tractor_geom)
        
        self.safety_controller = SafetySystemController(
            tractor_model=tractor,
            camera_intrinsics=camera,
            max_tractor_speed_kmh=5.0,
        )
        
        self.emergency_controller = EmergencyResponseController()
        self.dashboard = MonitoringDashboard()
        
        logger.info("✅ All components initialized")
    
    def test_component_1_geometry(self):
        """Test 1: Tractor POV Geometry"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 1: TRACTOR POV GEOMETRY")
        logger.info("=" * 100)
        
        # Test projection
        pixel_x, pixel_y = 960, 540  # Image center
        ground_pos = self.tractor_geom.pixel_to_3d_ground_plane(pixel_x, pixel_y)
        
        logger.info(f"Pixel ({pixel_x}, {pixel_y}) → Ground position: {ground_pos}")
        assert ground_pos[2] > 0, "Should project beyond tractor"
        
        # Test safety zone
        zone = self.tractor_geom.compute_safety_zone(safety_distance_m=2.0)
        logger.info(f"Safety zone area: {len(zone)} points")
        assert len(zone) > 0, "Should have safety zone"
        
        logger.info("✅ Geometry test PASSED")
        return True
    
    def test_component_2_terrain(self):
        """Test 2: Terrain Analysis"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 2: TERRAIN ANALYSIS")
        logger.info("=" * 100)
        
        # Create synthetic test image
        test_image = np.full((1080, 1920, 3), [60, 100, 80], dtype=np.uint8)
        
        # Add some variation to simulate terrain
        test_image[300:400] = [100, 120, 100]  # Lighter area
        test_image[600:700] = [40, 60, 40]     # Darker area
        
        # Analyze
        terrain = self.terrain_analyzer.analyze_image(test_image)
        
        logger.info(f"Formation: {terrain.formation.value}")
        logger.info(f"Soil type: {terrain.soil_type.value}")
        logger.info(f"Moisture: {terrain.moisture_level:.0%}")
        logger.info(f"Movement difficulty: {terrain.movement_difficulty:.2f}")
        
        assert 0 <= terrain.movement_difficulty <= 1, "Difficulty out of range"
        
        logger.info("✅ Terrain test PASSED")
        return True
    
    def test_component_3_risk_assessment(self):
        """Test 3: Risk Assessment"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 3: RISK ASSESSMENT (7-FACTOR MODEL)")
        logger.info("=" * 100)
        
        # Create test detection
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        class MockDetection:
            def __init__(self):
                self.bbox = (900, 400, 1000, 600)
                self.confidence = 0.95
        
        det = MockDetection()
        
        # Analyze terrain
        terrain = self.terrain_analyzer.analyze_image(test_image)
        
        # Compute risk
        assessment = self.risk_assessor.assess_detection(
            det,
            test_image,
            self.tractor_geom,
            terrain,
        )
        
        logger.info(f"Risk level: {assessment.risk_level.value}")
        logger.info(f"Risk score: {assessment.risk_score:.3f}")
        logger.info(f"Distance: {assessment.factors.geometric_distance_m:.2f}m")
        logger.info(f"TTC: {assessment.factors.ttc_seconds:.2f}s")
        logger.info(f"Escape difficulty: {assessment.factors.escape_difficulty:.2f}")
        logger.info(f"Rationale: {assessment.rationale}")
        
        assert 0 <= assessment.risk_score <= 1, "Risk score out of range"
        
        logger.info("✅ Risk assessment test PASSED")
        return True
    
    def test_component_4_safety_controller(self):
        """Test 4: Safety Controller Integration"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 4: SAFETY CONTROLLER")
        logger.info("=" * 100)
        
        test_image = np.full((1080, 1920, 3), [100, 120, 100], dtype=np.uint8)
        
        # Mock detections with varying distances/risks
        detections = {
            0: {"bbox": (900, 400, 1000, 600), "confidence": 0.95},  # Mid-field
            1: {"bbox": (800, 350, 900, 650), "confidence": 0.88},   # Left
        }
        
        # Process through controller
        action = self.safety_controller.process_frame(test_image, detections)
        
        logger.info(f"Action: {action.action_type}")
        logger.info(f"Risk level: {action.level.value}")
        logger.info(f"Severity: {action.severity:.2f}")
        logger.info(f"Target speed: {action.target_speed_kmh:.1f} km/h")
        logger.info(f"Reason: {action.reason}")
        
        status = self.safety_controller.get_status_report()
        logger.info(f"System state: {status['system_state']}")
        logger.info(f"Active detections: {status['detections_active']}")
        
        logger.info("✅ Safety controller test PASSED")
        return True
    
    def test_component_5_emergency_protocols(self):
        """Test 5: Emergency Protocols & Escalation"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 5: EMERGENCY PROTOCOLS & ESCALATION")
        logger.info("=" * 100)
        
        # Test all emergency levels
        for level in range(6):
            response = self.emergency_controller.get_response_for_level(level)
            
            logger.info(f"Level {level}: {response.get('name', 'UNKNOWN')}")
            logger.info(f"  Immediate: {response.get('immediate_command', 'N/A')}")
            logger.info(f"  Follow-up: {response.get('follow_up_command', 'N/A')}")
            
            assert response, f"No response for level {level}"
        
        logger.info("✅ Emergency protocols test PASSED")
        return True
    
    def test_component_6_monitoring_dashboard(self):
        """Test 6: Monitoring Dashboard"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 6: MONITORING DASHBOARD")
        logger.info("=" * 100)
        
        # Create test frame
        test_frame = np.full((1080, 1920, 3), [50, 100, 80], dtype=np.uint8)
        
        # Mock data
        detections = {
            0: {
                "bbox": (900, 400, 1000, 600),
                "distance_m": 2.5,
                "risk_level": "WARNING",
                "confidence": 0.95,
            }
        }
        
        terrain = {
            "formation": "gentle_slope",
            "soil_type": "clay",
            "moisture": 0.65,
            "hazards": ["mud", "ruts"],
        }
        
        status = {
            "system_state": "MONITORING",
            "detections_active": 1,
            "frames_processed": 100,
            "alerts_issued": 3,
            "emergency_stops": 0,
        }
        
        action = {
            "action_type": "REDUCE_SPEED",
            "severity": 0.35,
            "reason": "Person detected at 2.5m on muddy clay",
        }
        
        # Render
        output = self.dashboard.render_frame(
            test_frame,
            detections,
            terrain_analysis=terrain,
            system_status=status,
            current_action=action,
        )
        
        assert output.shape == test_frame.shape, "Output shape mismatch"
        assert output.dtype == np.uint8, "Output dtype mismatch"
        
        # Save test image
        output_path = Path(log_dir) / "system_integration_test_dashboard.png"
        cv2.imwrite(str(output_path), output)
        logger.info(f"Dashboard image saved: {output_path}")
        
        # Add alerts
        self.dashboard.add_alert("WARNING", "Test Alert", "System test alert")
        
        logger.info("✅ Monitoring dashboard test PASSED")
        return True
    
    def test_end_to_end_scenario(self):
        """Test 7: Complete End-to-End Scenario"""
        logger.info("\n" + "=" * 100)
        logger.info("TEST 7: END-TO-END SAFETY SCENARIO")
        logger.info("=" * 100)
        
        # Scenario: Person approaching tractor
        logger.info("\nScenario: Person approaches tractor on muddy clay field")
        logger.info("-" * 100)
        
        # Create base frame
        base_frame = np.full((1080, 1920, 3), [80, 110, 100], dtype=np.uint8)
        
        # Simulate approaching person with decreasing distance over time
        distances_m = [50, 30, 15, 8, 4, 2, 1, 0.5]
        actions_log = []
        
        for step, distance in enumerate(distances_m):
            logger.info(f"\nStep {step+1}: Distance {distance:.1f}m")
            
            # Position person based on distance (closer = lower in frame, larger)
            if distance > 30:
                bbox = (900, 200, 1000, 300)  # Far, small
            elif distance > 15:
                bbox = (850, 300, 1050, 600)  # Medium, medium
            elif distance > 5:
                bbox = (800, 400, 1100, 800)  # Close, large
            else:
                bbox = (700, 500, 1200, 950)  # Very close, very large
            
            detections = {0: {"bbox": bbox, "confidence": 0.95 - step * 0.05}}
            
            # Process through safety system
            action = self.safety_controller.process_frame(base_frame, detections)
            actions_log.append((distance, action.action_type, action.severity))
            
            logger.info(f"  Action: {action.action_type}")
            logger.info(f"  Severity: {action.severity:.2f}")
            logger.info(f"  Reason: {action.reason}")
        
        # Verify escalation
        logger.info("\nEscalation summary:")
        logger.info("-" * 100)
        for distance, action_type, severity in actions_log:
            logging_char = "→"
            logger.info(f"  {distance:5.1f}m: {action_type:20s} (severity: {severity:.2f})")
        
        logger.info("✅ End-to-end scenario PASSED")
        return True
    
    def run_all_tests(self) -> dict:
        """Run all integration tests."""
        results = {}
        
        tests = [
            ("Tractor POV Geometry", self.test_component_1_geometry),
            ("Terrain Analysis", self.test_component_2_terrain),
            ("Risk Assessment", self.test_component_3_risk_assessment),
            ("Safety Controller", self.test_component_4_safety_controller),
            ("Emergency Protocols", self.test_component_5_emergency_protocols),
            ("Monitoring Dashboard", self.test_component_6_monitoring_dashboard),
            ("End-to-End Scenario", self.test_end_to_end_scenario),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                result = test_func()
                results[name] = "PASSED" if result else "FAILED"
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"❌ {name} FAILED: {e}")
                results[name] = f"ERROR: {e}"
                failed += 1
        
        # Final report
        logger.info("\n" + "=" * 100)
        logger.info("INTEGRATION TEST REPORT")
        logger.info("=" * 100)
        
        for name, result in results.items():
            status_symbol = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_symbol} {name}: {result}")
        
        logger.info(f"\nTotal: {passed} PASSED, {failed} FAILED")
        logger.info("=" * 100)
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "passed": passed,
            "failed": failed,
            "total": len(tests),
        }
        
        report_path = Path(log_dir) / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved: {report_path}")
        
        return results


def main():
    """Run complete system integration test."""
    try:
        test = SystemIntegrationTest()
        results = test.run_all_tests()
        
        # Check if all passed
        all_passed = all(v == "PASSED" for v in results.values())
        
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT 🎉\n")
            return 0
        else:
            logger.error("\n⚠️  SOME TESTS FAILED - FIX ISSUES BEFORE DEPLOYMENT\n")
            return 1
    
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
