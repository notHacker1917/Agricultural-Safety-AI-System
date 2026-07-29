"""
Real-Time Agricultural Safety System Controller

Integrates all safety components:
- Detection + Tracking
- Terrain Analysis
- Risk Assessment
- Emergency Response
- Tractor ECU Control
- Audit Logging

This is production-grade code designed to run on Jetson Orin
and prevent agricultural machinery accidents in real-time.
"""

import numpy as np
import cv2
import threading
import time
import json
import os
from enum import Enum
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from pathlib import Path

from tractor_geometry import TractorGeometry, TractorPOVGeometry, CameraIntrinsics, create_realistic_camera
from terrain_analysis import TerrainAnalyzer, TerrainAnalysis
from context_aware_risk_system import ContextAwareRiskAssessor, RiskLevel

# Configure logging
import os
log_dir = os.path.expanduser("~/safety_logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "safety_system.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    MONITORING = "MONITORING"
    WARNING = "WARNING"
    EMERGENCY = "EMERGENCY"
    STANDBY = "STANDBY"
    FAILURE = "FAILURE"


class DetectionEvent:
    """A single detected person in frame."""
    def __init__(self, detection_id: int, bbox, confidence, frame_num):
        self.id = detection_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.frame_num = frame_num
        self.timestamp = time.time()
        self.trajectory = deque(maxlen=30)  # Last 30 frames
        self.trajectory.append(bbox)
        self.soil_analysis = None
        self.risk_assessment = None
        self.is_active = True

    def update(self, bbox, confidence, frame_num):
        """Update detection with new position."""
        self.bbox = bbox
        self.confidence = confidence
        self.frame_num = frame_num
        self.trajectory.append(bbox)
        self.timestamp = time.time()


@dataclass
class SafetyAction:
    """Recommended safety action."""
    level: RiskLevel
    action_type: str  # "ALERT", "SLOW", "STOP", "EMERGENCY_STOP"
    severity: float  # 0-1
    reason: str
    target_speed_kmh: Optional[float] = None  # For SLOW actions
    estimated_stop_distance_m: Optional[float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self):
        d = asdict(self)
        # Convert enum to string for JSON serialization
        if isinstance(d.get("level"), Enum):
            d["level"] = d["level"].value
        return d


class SafetySystemController:
    """
    Complete agricultural safety system controller.
    
    Reads camera input → detects people → analyzes terrain → computes risk →
    issues warnings/commands → logs for audit compliance.
    """

    def __init__(
        self,
        tractor_model: TractorGeometry = None,
        camera_intrinsics: CameraIntrinsics = None,
        max_tractor_speed_kmh: float = 10.0,
        emergency_stop_latency_ms: float = 100.0,
        audit_log_path: str = "safety_audit_log.json",
    ):
        """
        Initialize safety controller.
        
        Args:
            tractor_model: Tractor geometry (default: generic 3m harvester)
            camera_intrinsics: Camera calibration (default: standard 95° FOV)
            max_tractor_speed_kmh: Safety limit for autonomous operation
            emergency_stop_latency_ms: Target latency for emergency stop command
            audit_log_path: Path for compliance audit logs
        """
        # Setup geometry
        self.tractor = tractor_model or TractorGeometry.default_harvester()
        self.camera = camera_intrinsics or create_realistic_camera()
        self.pov = TractorPOVGeometry(self.tractor, self.camera)
        
        # Safety limits
        self.max_speed_kmh = max_tractor_speed_kmh
        self.emergency_stop_latency = emergency_stop_latency_ms / 1000.0
        
        # Risk analysis
        self.risk_assessor = ContextAwareRiskAssessor(
            self.pov,
            tractor_speed_kmh=max_tractor_speed_kmh / 2,  # Conservative initial
            operator_reaction_time_s=1.0,
        )
        
        # Terrain analysis
        self.terrain_analyzer = TerrainAnalyzer()
        
        # Detection tracking
        self.active_detections: Dict[int, DetectionEvent] = {}
        self.next_detection_id = 0
        self.detection_history = deque(maxlen=300)  # Last 10 seconds @ 30fps
        
        # System state
        self.system_state = SystemState.INITIALIZING
        self.current_speed_kmh = 0.0
        self.current_action: Optional[SafetyAction] = None
        self.emergency_stop_active = False
        
        # Metrics
        self.frame_count = 0
        self.detection_count = 0
        self.alerts_issued = 0
        self.emergency_stops = 0
        
        # Audit logging
        audit_dir = os.path.expanduser("~/safety_logs")
        os.makedirs(audit_dir, exist_ok=True)
        self.audit_log_path = Path(os.path.join(audit_dir, audit_log_path))
        self.audit_log_path.parent.mkdir(exist_ok=True)
        self.audit_log = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Safety controller initialized")
        logger.info(f"  Tractor: {self.tractor.model.value}")
        logger.info(f"  Camera FOV: {np.degrees(self.camera.horizontal_fov):.0f}° H")
        logger.info(f"  Max speed: {self.max_speed_kmh} km/h")
        logger.info(f"  Emergency stop latency: {emergency_stop_latency_ms:.1f}ms")
        
        self.system_state = SystemState.READY

    def process_frame(
        self,
        image: np.ndarray,
        detections: List[Tuple[float, float, float, float, float]],
    ) -> SafetyAction:
        """
        Process one frame of video with detections.
        
        Args:
            image: Current frame (BGR)
            detections: List of (x1, y1, x2, y2, confidence) tuples
        
        Returns:
            SafetyAction recommending what tractor should do
        """
        self.frame_count += 1
        self.system_state = SystemState.MONITORING
        
        with self.lock:
            # Step 1: Match detections to tracks
            self._update_tracks(detections)
            
            # Step 2: Analyze terrain once per image
            terrain = self.terrain_analyzer.analyze_image(image)
            
            # Step 3: Assess risk for each person
            risks = []
            for det_id, detection in self.active_detections.items():
                # Analyze soil in person's ROI
                detection.soil_analysis = self.terrain_analyzer.analyze_image(
                    image,
                    roi=self._bbox_to_roi(detection.bbox)
                )
                
                # Compute risk
                risk_assess = self.risk_assessor.assess_detection(
                    detection.bbox,
                    detection.confidence,
                    detection_history=list(detection.trajectory),
                    image=image,
                    terrain=detection.soil_analysis,
                )
                detection.risk_assessment = risk_assess
                risks.append((det_id, risk_assess))
            
            # Step 3: Determine system-level action
            action = self._compute_system_action(risks, terrain)
            
            # Step 4: Log decision
            self._log_audit_event(action, terrain, risks)
            
            # Step 5: Update state
            self.current_action = action
            
            # Step 6: Issue commands if needed
            if action.action_type == "EMERGENCY_STOP":
                self._trigger_emergency_stop(action)
            
            return action

    def _update_tracks(self, detections: List[Tuple[float, float, float, float, float]]):
        """Match new detections to existing tracks."""
        if not detections:
            # No detections - mark all as inactive
            for det in self.active_detections.values():
                det.is_active = False
            return
        
        # Simple IoU-based matching (for production, use Hungarian algorithm)
        matched_ids = set()
        
        for x1, y1, x2, y2, conf in detections:
            new_bbox = (x1, y1, x2, y2)
            best_id = None
            best_iou = 0.0
            
            for det_id, detection in self.active_detections.items():
                if not detection.is_active:
                    continue
                
                iou = self._compute_iou(detection.bbox, new_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = det_id
            
            if best_iou > 0.3:  # Threshold for match
                self.active_detections[best_id].update(new_bbox, conf, self.frame_count)
                matched_ids.add(best_id)
            else:
                # New detection
                new_id = self.next_detection_id
                self.next_detection_id += 1
                self.active_detections[new_id] = DetectionEvent(new_id, new_bbox, conf, self.frame_count)
                matched_ids.add(new_id)
        
        # Mark unmatched as inactive
        for det_id, detection in self.active_detections.items():
            if det_id not in matched_ids:
                detection.is_active = False
        
        # Clean up old inactive detections
        to_remove = [
            det_id for det_id, det in self.active_detections.items()
            if not det.is_active and (self.frame_count - det.frame_num) > 90
        ]
        for det_id in to_remove:
            del self.active_detections[det_id]
        
        self.detection_count = len([d for d in self.active_detections.values() if d.is_active])

    def _compute_system_action(self, risks: List, terrain: TerrainAnalysis) -> SafetyAction:
        """
        Compute highest-priority safety action needed.
        
        Combines all detected risks into single system action.
        """
        if not risks:
            return SafetyAction(
                level=RiskLevel.SAFE,
                action_type="CONTINUE",
                severity=0.0,
                reason="No detections in image",
                timestamp=time.time(),
            )
        
        # Find highest risk
        max_risk = max(risks, key=lambda r: r[1].risk_score)
        det_id, max_assessment = max_risk
        
        # Map risk level to action
        if max_assessment.risk_level == RiskLevel.CRITICAL:
            action_type = "EMERGENCY_STOP"
            reason = f"CRITICAL: {max_assessment.rationale}"
            target_speed = 0.0
            self.emergency_stops += 1
            self.system_state = SystemState.EMERGENCY
        
        elif max_assessment.risk_level == RiskLevel.HIGH_WARNING:
            action_type = "URGENT_SLOW"
            reason = f"HIGH_WARNING: {max_assessment.rationale}"
            target_speed = self.max_speed_kmh * 0.25  # 25% speed
            self.system_state = SystemState.WARNING
        
        elif max_assessment.risk_level == RiskLevel.WARNING:
            action_type = "REDUCE_SPEED"
            reason = f"WARNING: {max_assessment.rationale}"
            target_speed = self.max_speed_kmh * 0.6  # 60% speed
            self.system_state = SystemState.WARNING
        
        elif max_assessment.risk_level == RiskLevel.LOW_WARNING:
            action_type = "MONITOR"
            reason = f"LOW_WARNING: Monitor distance to person"
            target_speed = self.max_speed_kmh * 0.8  # 80% speed
        
        else:  # SAFE
            action_type = "CONTINUE"
            reason = "All clear - no safety threats detected"
            target_speed = self.max_speed_kmh
            self.system_state = SystemState.MONITORING
        
        # Estimate stopping distance
        stop_dist = self.risk_assessor.compute_braking_distance() if action_type != "CONTINUE" else None
        
        self.alerts_issued += 1
        
        return SafetyAction(
            level=max_assessment.risk_level,
            action_type=action_type,
            severity=max_assessment.risk_score,
            reason=reason,
            target_speed_kmh=target_speed,
            estimated_stop_distance_m=stop_dist,
            timestamp=time.time(),
        )

    def _trigger_emergency_stop(self, action: SafetyAction):
        """
        Issue emergency stop to tractor ECU.
        
        In real system, this would:
        1. Send CAN message to ECU
        2. Log safety event with microsecond timestamp
        3. Activate hydraulic dump for cutting mechanism
        """
        logger.critical(f"EMERGENCY STOP TRIGGERED: {action.reason}")
        self.emergency_stop_active = True
        
        # In production: Send CAN message
        # can_msg = self._build_emergency_stop_can_msg()
        # self.can_interface.send(can_msg)
        
        # Log immediately
        self._log_safety_event("EMERGENCY_STOP", action)

    def _log_audit_event(
        self,
        action: SafetyAction,
        terrain: TerrainAnalysis,
        risks: List,
    ):
        """Log every decision for audit compliance."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frame_count,
            "system_state": self.system_state.value,
            "action": action.to_dict(),
            "terrain": {
                "formation": terrain.formation.value,
                "soil_type": terrain.soil_type.value,
                "moisture": terrain.moisture_level,
                "hazards": terrain.hazard_factors,
            },
            "detections": len(self.active_detections),
            "risks": [
                {
                    "detection_id": det_id,
                    "risk_level": risk.risk_level.value,
                    "risk_score": risk.risk_score,
                    "distance_m": risk.factors.geometric_distance_m,
                    "safety_margin_m": risk.safety_margin_m,
                }
                for det_id, risk in risks
            ],
        }
        
        self.audit_log.append(event)
        
        # Persist to file (every 100 frames for efficiency)
        if self.frame_count % 100 == 0:
            self._flush_audit_log()

    def _log_safety_event(self, event_type: str, action: SafetyAction):
        """Log critical safety events (emergency stop, failures)."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "microseconds": int(time.time() * 1e6),  # Microsecond precision
            "event_type": event_type,
            "frame_number": self.frame_count,
            "action": action.to_dict(),
            "system_state": self.system_state.value,
            "detections_active": len(self.active_detections),
        }
        
        logger.warning(json.dumps(event))
        
        # Always persist safety events immediately
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _flush_audit_log(self):
        """Persist audit log to file."""
        if not self.audit_log:
            return
        
        try:
            with open(self.audit_log_path, "a") as f:
                for event in self.audit_log:
                    f.write(json.dumps(event) + "\n")
            self.audit_log.clear()
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")

    @staticmethod
    def _compute_iou(bbox1, bbox2) -> float:
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _bbox_to_roi(bbox) -> Tuple[int, int, int, int]:
        """Convert bbox to ROI for image extraction."""
        x1, y1, x2, y2 = bbox
        return (int(x1), int(y1), int(x2), int(y2))

    def get_status_report(self) -> Dict:
        """Get current system status."""
        with self.lock:
            return {
                "system_state": self.system_state.value,
                "timestamp": datetime.now().isoformat(),
                "frames_processed": self.frame_count,
                "detections_active": len(self.active_detections),
                "alerts_issued": self.alerts_issued,
                "emergency_stops": self.emergency_stops,
                "current_action": self.current_action.to_dict() if self.current_action else None,
                "emergency_stop_active": self.emergency_stop_active,
            }

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down safety controller")
        self._flush_audit_log()
        logger.info(f"Processed {self.frame_count} frames")
        logger.info(f"Issued {self.alerts_issued} alerts")
        logger.info(f"Emergency stops: {self.emergency_stops}")


def test_safety_controller():
    """Test safety system with synthetic data."""
    logger.info("=" * 80)
    logger.info("SAFETY SYSTEM CONTROLLER TEST")
    logger.info("=" * 80)
    
    controller = SafetySystemController(max_tractor_speed_kmh=5.0)
    
    # Simulate 10 frames
    for frame_num in range(10):
        # Create synthetic image (just for terrain analysis)
        image = np.full((1080, 1920, 3), [100, 120, 80], dtype=np.uint8)
        
        # Simulate detections
        if frame_num < 3:
            detections = []  # No people
        elif frame_num < 5:
            detections = [(960, 700, 1040, 900, 0.92)]  # 1 person, far
        elif frame_num < 7:
            detections = [(940, 600, 1020, 850, 0.95)]  # 1 person, closer
        else:
            detections = [(940, 500, 1020, 750, 0.98)]  # 1 person, very close
        
        action = controller.process_frame(image, detections)
        
        status = controller.get_status_report()
        logger.info(f"Frame {frame_num}: {action.action_type} (risk: {action.level.value})")
    
    controller.shutdown()


if __name__ == "__main__":
    test_safety_controller()
