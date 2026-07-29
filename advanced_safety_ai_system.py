#!/usr/bin/env python3
"""
Advanced Agricultural Safety AI System with LLM Integration

Combines multiple detection algorithms, LLM-powered risk assessment,
and advanced safety protocols for comprehensive human detection in agricultural environments.

Features:
- Multi-scale human detection
- Motion-based tracking
- Depth estimation and risk categorization
- Contextual awareness with temporal analysis
- LLM-enhanced risk assessment with safety-critical decision making
- Real-time visualization and alerts
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import argparse
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import queue

# Import our advanced components
from advanced_detection_algorithms import EnsembleHumanDetector
from advanced_llm_risk_assessor import (
    AdvancedLLMRiskAssessor,
    HumanDetectionInput,
    RiskAssessmentOutput,
    RiskLevel,
    assess_human_risk,
    create_risk_assessor
)
from advanced_trajectory_predictor import (
    AdvancedTrajectoryPredictor,
    PredictedTrajectory,
    predict_human_trajectory
)
from detection import ObjectDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SafetyAlert:
    """Safety alert with risk assessment."""
    timestamp: float
    object_id: str
    risk_level: RiskLevel
    risk_score: float
    position: Tuple[float, float]
    reason: str
    recommended_action: str

class AdvancedSafetyAISystem:
    """
    Complete agricultural safety AI system with LLM integration.

    Combines:
    1. Advanced multi-algorithm human detection
    2. LLM-powered risk assessment
    3. Real-time safety monitoring
    4. Emergency response protocols
    """

    def __init__(self,
                 llm_provider: str = "mock",
                 safety_zone_radius: float = 0.3,
                 emergency_threshold: RiskLevel = RiskLevel.HIGH,
                 enable_visualization: bool = True):
        """
        Initialize the complete safety AI system.

        Args:
            llm_provider: LLM provider for risk assessment
            safety_zone_radius: Normalized safety zone radius
            emergency_threshold: Risk level that triggers emergency protocols
            enable_visualization: Whether to show real-time visualization
        """
        self.llm_provider = llm_provider
        self.safety_zone_radius = safety_zone_radius
        self.emergency_threshold = emergency_threshold
        self.enable_visualization = enable_visualization

        # Initialize detection system
        logger.info("Initializing advanced detection system...")
        self.base_detector = ObjectDetector('yolov8n.pt', conf=0.5)
        self.ensemble_detector = EnsembleHumanDetector(
            base_yolo_detector=self.base_detector,
            use_motion=True,
            use_depth=True,
            use_context=True
        )

        # Initialize risk assessment system
        logger.info("Initializing LLM-enhanced risk assessment...")
        self.risk_assessor = create_risk_assessor(
            llm_provider=llm_provider,
            safety_zone_radius=safety_zone_radius
        )

        # Initialize advanced trajectory prediction system
        logger.info("Initializing advanced trajectory prediction...")
        self.trajectory_predictor = AdvancedTrajectoryPredictor(
            use_lstm=True, use_physics=True, use_llm=True
        )

        # Safety monitoring
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.alert_history: List[SafetyAlert] = []
        self.emergency_active = False

        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.detection_stats = defaultdict(int)

        # Threading for real-time processing
        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()

        logger.info("Advanced Safety AI System initialized successfully")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through the complete safety pipeline.

        Args:
            frame: Input video frame

        Returns:
            Processing results with detections, risks, and alerts
        """
        start_time = time.time()

        try:
            # Step 1: Advanced human detection
            detections = self.ensemble_detector.detect(frame)

            # Step 2: Risk assessment for each detection
            risk_assessments = []
            trajectory_predictions = []
            alerts = []

            # Define tractor position (center-bottom of frame)
            frame_h, frame_w = frame.shape[:2]
            tractor_position = (frame_w / 2, frame_h * 0.9)

            for detection in detections:
                # Convert detection to risk assessment input
                risk_input = self._detection_to_risk_input(detection, frame.shape)

                # Assess risk
                risk_result = self.risk_assessor.assess_risk(risk_input)
                risk_dict = asdict(risk_result)
                risk_dict['risk_level'] = risk_result.risk_level.value  # Convert enum to string
                risk_assessments.append({
                    'detection': detection,
                    'risk': risk_dict
                })

                # Step 2.5: Advanced trajectory prediction
                object_id = detection.get('object_id', f"obj_{self.frame_count}_{len(trajectory_predictions)}")
                predicted_trajectory = self.trajectory_predictor.predict_trajectory(
                    object_id, detection, frame, tractor_position, time_horizon=30
                )
                trajectory_predictions.append({
                    'object_id': object_id,
                    'trajectory': predicted_trajectory
                })

                # Enhanced risk assessment using trajectory prediction
                trajectory_risk = predicted_trajectory.risk_assessment
                if trajectory_risk['risk_level'] in ['HIGH', 'CRITICAL']:
                    # Create trajectory-based alert
                    alert = self._create_trajectory_alert(detection, trajectory_risk, predicted_trajectory)
                    alerts.append(alert)
                    self.active_alerts[object_id] = alert

                # Check for alerts
                if self._should_alert(risk_result):
                    alert = self._create_alert(detection, risk_result)
                    alerts.append(alert)
                    self.active_alerts[detection.get('object_id', 'unknown')] = alert

            # Step 3: Update emergency status
            self._update_emergency_status(alerts)

            # Step 4: Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1

            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]

            # Update detection statistics
            for assessment in risk_assessments:
                risk_level = assessment['risk']['risk_level']
                self.detection_stats[risk_level] += 1

            result = {
                'frame_number': self.frame_count,
                'processing_time': processing_time,
                'detections': detections,
                'risk_assessments': risk_assessments,
                'trajectory_predictions': trajectory_predictions,
                'alerts': alerts,
                'emergency_active': self.emergency_active,
                'system_status': self.get_system_status()
            }

            return result

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                'error': str(e),
                'emergency_active': True,  # Safety-first on errors
                'system_status': self.get_system_status()
            }

    def _detection_to_risk_input(self, detection: Dict[str, Any],
                               frame_shape: Tuple[int, int, int]) -> HumanDetectionInput:
        """
        Convert detection result to risk assessment input format.
        """
        # Extract position (center of bbox)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Normalize position to frame coordinates
        frame_h, frame_w = frame_shape[:2]
        norm_x = center_x / frame_w
        norm_y = center_y / frame_h

        # Calculate distance to tractor (assuming tractor at center-bottom)
        tractor_x, tractor_y = frame_w / 2, frame_h * 0.9  # Tractor position assumption
        distance_pixels = np.sqrt((center_x - tractor_x)**2 + (center_y - tractor_y)**2)
        distance_normalized = min(distance_pixels / (frame_w / 2), 1.0)  # Normalize

        # Estimate velocity (simplified - would need tracking)
        velocity = detection.get('velocity', (0, 0))
        speed = detection.get('speed', np.linalg.norm(velocity))

        # Determine direction toward tractor
        dx = tractor_x - center_x
        dy = tractor_y - center_y
        direction_toward = (dx * velocity[0] + dy * velocity[1]) > 0 if velocity != (0, 0) else False

        # Predicted path (simplified linear extrapolation)
        predicted_path = []
        if velocity != (0, 0):
            for i in range(1, 6):  # 5 future positions
                future_x = center_x + velocity[0] * i * 5  # 5 frames ahead
                future_y = center_y + velocity[1] * i * 5
                predicted_path.append((future_x, future_y))

        # Safety zone check
        will_enter_safety_zone = distance_normalized <= self.safety_zone_radius * 1.5

        # Time to collision estimate
        time_to_collision = None
        if speed > 0.1 and direction_toward:
            time_to_collision = distance_pixels / speed

        return HumanDetectionInput(
            object_id=detection.get('object_id', f"obj_{self.frame_count}"),
            current_position=(norm_x, norm_y),
            distance_to_tractor=distance_normalized,
            velocity=velocity,
            speed=speed,
            direction_toward_tractor=direction_toward,
            predicted_path=predicted_path,
            will_enter_safety_zone=will_enter_safety_zone,
            time_to_collision=time_to_collision,
            is_occluded=detection.get('is_occluded', False),
            detection_confidence=detection.get('confidence', 0.5)
        )

    def _should_alert(self, risk_result: RiskAssessmentOutput) -> bool:
        """
        Determine if a risk assessment should trigger an alert.
        """
        return risk_result.risk_level.value in ['HIGH', 'CRITICAL']

    def _create_alert(self, detection: Dict[str, Any],
                     risk_result: RiskAssessmentOutput) -> SafetyAlert:
        """
        Create a safety alert from detection and risk assessment.
        """
        # Determine recommended action based on risk level
        actions = {
            'HIGH': 'Slow down machinery and monitor closely',
            'CRITICAL': 'IMMEDIATE STOP - Human in danger zone'
        }

        return SafetyAlert(
            timestamp=time.time(),
            object_id=detection.get('object_id', 'unknown'),
            risk_level=risk_result.risk_level,
            risk_score=risk_result.risk_score,
            position=tuple(detection.get('bbox', [0, 0, 0, 0])[:2]),
            reason=risk_result.reason,
            recommended_action=actions.get(risk_result.risk_level.value, 'Monitor situation')
        )

    def _create_trajectory_alert(self, detection: Dict[str, Any],
                               trajectory_risk: Dict[str, Any],
                               predicted_trajectory: PredictedTrajectory) -> SafetyAlert:
        """
        Create a safety alert based on trajectory prediction risk.
        """
        # Determine recommended action based on trajectory risk
        risk_level_str = trajectory_risk['risk_level']
        actions = {
            'HIGH': 'Trajectory indicates potential danger - prepare to stop',
            'CRITICAL': 'EMERGENCY: Predicted trajectory shows imminent collision risk!'
        }

        # Enhanced reason with trajectory information
        reason = f"Trajectory prediction: {trajectory_risk['risk_level']} risk. "
        reason += f"Min distance: {trajectory_risk['min_distance']:.1f}px. "
        reason += f"Environmental: {predicted_trajectory.environmental_conditions.get('weather', 'unknown')}"

        return SafetyAlert(
            timestamp=time.time(),
            object_id=detection.get('object_id', 'unknown'),
            risk_level=RiskLevel[risk_level_str],
            risk_score=trajectory_risk['risk_score'],
            position=tuple(detection.get('bbox', [0, 0, 0, 0])[:2]),
            reason=reason,
            recommended_action=actions.get(risk_level_str, 'Monitor predicted trajectory')
        )

    def _update_emergency_status(self, alerts: List[SafetyAlert]):
        """
        Update emergency status based on active alerts.
        """
        critical_alerts = [a for a in alerts if a.risk_level == RiskLevel.CRITICAL]
        self.emergency_active = len(critical_alerts) > 0

        if self.emergency_active:
            logger.warning("🚨 EMERGENCY PROTOCOL ACTIVATED - Critical risk detected!")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        """
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        return {
            'frames_processed': self.frame_count,
            'average_processing_time': round(avg_processing_time, 3),
            'fps': round(1.0 / avg_processing_time, 1) if avg_processing_time > 0 else 0,
            'active_alerts': len(self.active_alerts),
            'emergency_active': self.emergency_active,
            'detection_stats': dict(self.detection_stats),
            'risk_assessor_status': self.risk_assessor.get_system_status()
        }

    def reset_system(self):
        """
        Reset system state for new session.
        """
        self.active_alerts.clear()
        self.alert_history.clear()
        self.emergency_active = False
        self.frame_count = 0
        self.processing_times.clear()
        self.detection_stats.clear()
        self.risk_assessor.reset_trajectory_history()
        logger.info("System reset complete")

def run_live_demo(system: AdvancedSafetyAISystem,
                  input_type: str = 'webcam',
                  input_path: Optional[str] = None,
                  max_frames: int = 100):
    """
    Run live demonstration of the advanced safety AI system.
    """
    logger.info("🚜 Starting Advanced Agricultural Safety AI Demo")
    logger.info("=" * 60)

    # Initialize video capture
    if input_type == 'webcam':
        cap = cv2.VideoCapture(0)
        logger.info("Using webcam input")
    elif input_type == 'video' and input_path:
        cap = cv2.VideoCapture(input_path)
        logger.info(f"Using video input: {input_path}")
    else:
        logger.error("Invalid input type or path")
        return

    if not cap.isOpened():
        logger.error("Could not open video source")
        return

    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < max_frames and not system.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame through safety system
            result = system.process_frame(frame)

            # Display results
            display_frame = frame.copy()

            # Draw detections and risk levels
            for assessment in result.get('risk_assessments', []):
                detection = assessment['detection']
                risk = assessment['risk']

                bbox = detection.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)

                # Color based on risk level
                colors = {
                    'SAFE': (0, 255, 0),        # Green
                    'LOW': (0, 255, 255),       # Light yellow
                    'MEDIUM': (0, 255, 255),    # Yellow
                    'HIGH': (0, 165, 255),      # Orange
                    'CRITICAL': (0, 0, 255)     # Red
                }

                color = colors.get(risk['risk_level'], (255, 255, 255))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Add risk information
                label = f"{risk['risk_level']} ({risk['risk_score']:.2f})"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add system status overlay
            status = result.get('system_status', {})
            status_text = [
                f"FPS: {status.get('fps', 0):.1f}",
                f"Alerts: {status.get('active_alerts', 0)}",
                f"Emergency: {'YES' if result.get('emergency_active', False) else 'NO'}"
            ]

            for i, text in enumerate(status_text):
                cv2.putText(display_frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add functional 3x3 grid overlay for better detection guidance
            height, width = display_frame.shape[:2]

            # Define tractor/equipment position (center-bottom of frame)
            tractor_x, tractor_y = width // 2, int(height * 0.85)

            # Draw 3x3 grid lines (4 vertical, 4 horizontal lines creating 9 equal squares)
            grid_color = (255, 255, 255)  # White
            grid_alpha = 0.02  # Very light overlay

            # Vertical grid lines (every 1/3 of width)
            for i in range(1, 3):  # 2 lines creating 3 columns
                x = int(width * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Horizontal grid lines (every 1/3 of height)
            for i in range(1, 3):  # 2 lines creating 3 rows
                y = int(height * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (0, y), (width, y), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Draw tractor/equipment marker
            cv2.circle(display_frame, (tractor_x, tractor_y), 12, (255, 0, 255), 2)  # Smaller, thinner magenta circle
            cv2.putText(display_frame, "TRACTOR", (tractor_x - 25, tractor_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # Add grid zone labels (3x3 grid zones)
            zone_labels = [
                ("NW", (width//6, height//6)),
                ("N", (width//2, height//6)),
                ("NE", (5*width//6, height//6)),
                ("W", (width//6, height//2)),
                ("CENTER", (width//2, height//2)),
                ("E", (5*width//6, height//2)),
                ("SW", (width//6, 5*height//6)),
                ("S", (width//2, 5*height//6)),
                ("SE", (5*width//6, 5*height//6))
            ]

            for label, pos in zone_labels:
                cv2.putText(display_frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            # Show frame
            cv2.imshow('Advanced Agricultural Safety AI', display_frame)

            frame_count += 1

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        final_status = system.get_system_status()

        logger.info("\n🎯 Demo Complete - Final Statistics:")
        logger.info(f"   Frames processed: {frame_count}")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Average FPS: {frame_count / total_time:.1f}")
        logger.info(f"   Risk distribution: {final_status['detection_stats']}")
        logger.info(f"   Emergency events: {final_status['emergency_active']}")

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Advanced Agricultural Safety AI System')
    parser.add_argument('--input-type', choices=['webcam', 'video'],
                       default='webcam', help='Input type')
    parser.add_argument('--input-path', help='Path to video file (for video input)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames to process')
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'mock'],
                       default='mock', help='LLM provider for risk assessment')
    parser.add_argument('--safety-zone', type=float, default=0.3,
                       help='Normalized safety zone radius')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    # Create safety system
    system = AdvancedSafetyAISystem(
        llm_provider=args.llm_provider,
        safety_zone_radius=args.safety_zone,
        enable_visualization=not args.no_viz
    )

    # Run demo
    run_live_demo(
        system=system,
        input_type=args.input_type,
        input_path=args.input_path,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()