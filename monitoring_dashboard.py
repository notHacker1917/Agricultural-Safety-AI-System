"""
Real-Time Monitoring & Alert System

Provides operator with real-time safety status,
visual alerts, and decision explanations.
"""

import threading
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert visibility levels."""
    INFO = "INFO"              # Informational
    WARNING = "WARNING"        # Caution
    CRITICAL = "CRITICAL"     # Immediate action needed
    EMERGENCY = "EMERGENCY"   # System failure


class VisualAlert:
    """Visual alert rendered on screen."""
    
    def __init__(
        self,
        alert_level: str,
        title: str,
        message: str,
        duration_seconds: float = 5.0,
    ):
        self.level = alert_level
        self.title = title
        self.message = message
        self.duration = duration_seconds
        self.created_at = time.time()
        self.is_active = True
    
    def update(self) -> bool:
        """Check if alert should still be shown."""
        elapsed = time.time() - self.created_at
        self.is_active = elapsed < self.duration
        return self.is_active
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get BGR color for alert level."""
        colors = {
            "INFO": (200, 200, 0),      # Cyan
            "WARNING": (0, 165, 255),   # Orange
            "CRITICAL": (0, 0, 255),    # Red
            "EMERGENCY": (0, 0, 255),   # Red (blinking)
        }
        return colors.get(self.level, (255, 255, 255))


class AlertQueue:
    """Manages alert queue with priority."""
    
    def __init__(self, max_alerts: int = 10):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.priority_levels = {
            "EMERGENCY": 0,
            "CRITICAL": 1,
            "WARNING": 2,
            "INFO": 3,
        }
        self.lock = threading.Lock()
    
    def add_alert(self, alert: VisualAlert):
        """Add alert to queue."""
        with self.lock:
            self.alerts.append(alert)
    
    def get_active_alerts(self) -> List[VisualAlert]:
        """Get currently active alerts sorted by priority."""
        with self.lock:
            active = [a for a in self.alerts if a.update()]
            return sorted(active, key=lambda a: self.priority_levels.get(a.level, 999))
    
    def clear_level(self, level: str):
        """Clear all alerts of specific level."""
        with self.lock:
            self.alerts = deque((a for a in self.alerts if a.level != level), maxlen=self.alerts.maxlen)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard rendered on video stream.
    Shows:
    - Person detections with distance labels
    - Terrain analysis
    - Risk assessment
    - System status
    - Recommended actions
    """
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        self.width = image_width
        self.height = image_height
        self.alert_queue = AlertQueue()
        
        # Fonts
        self.font_large = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.info(f"Monitoring dashboard initialized ({image_width}x{image_height})")
    
    def render_frame(
        self,
        image: np.ndarray,
        detections: Dict,
        terrain_analysis: Optional[Dict] = None,
        system_status: Optional[Dict] = None,
        current_action: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Render monitoring overlay on video frame.
        
        Args:
            image: Input frame (BGR)
            detections: Dict of {id: {bbox, distance, risk_level, ...}}
            terrain_analysis: Terrain info dict
            system_status: System status dict
            current_action: Current safety action
        
        Returns:
            Annotated frame ready for display
        """
        output = image.copy()
        
        # Render elements bottom-up (so text doesn't overlap)
        self._render_system_status(output, system_status)
        self._render_current_action(output, current_action)
        self._render_terrain_info(output, terrain_analysis)
        self._render_detections(output, detections)
        self._render_safety_zones(output)
        self._render_alerts(output)
        
        return output
    
    def _render_detections(self, image: np.ndarray, detections: Dict):
        """Render bounding boxes and distance labels."""
        if not detections:
            return
        
        for det_id, det_info in detections.items():
            x1, y1, x2, y2 = det_info.get("bbox", (0, 0, 10, 10))
            distance = det_info.get("distance_m", 0)
            risk_level = det_info.get("risk_level", "SAFE")
            confidence = det_info.get("confidence", 0.0)
            
            # Color based on risk
            color_map = {
                "CRITICAL": (0, 0, 255),
                "HIGH_WARNING": (0, 100, 255),
                "WARNING": (0, 165, 255),
                "LOW_WARNING": (0, 255, 255),
                "SAFE": (0, 255, 0),
            }
            color = color_map.get(risk_level, (255, 255, 255))
            
            # Draw bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw distance label
            label = f"ID{det_id} D:{distance:.1f}m {risk_level}"
            cv2.putText(
                image,
                label,
                (int(x1), int(y1) - 10),
                self.font_small,
                0.6,
                color,
                2,
            )
            
            # Draw confidence
            conf_label = f"Conf:{confidence:.0%}"
            cv2.putText(
                image,
                conf_label,
                (int(x1), int(y2) + 20),
                self.font_small,
                0.5,
                color,
                1,
            )
    
    def _render_safety_zones(self, image: np.ndarray):
        """Render tractor danger zones on image."""
        h, w = image.shape[:2]
        
        # Front danger zone (approximately bottom 25% of image)
        danger_zone_y = int(h * 0.75)
        cv2.rectangle(image, (0, danger_zone_y), (w, h), (0, 0, 255), 3)
        cv2.putText(
            image,
            "DANGER ZONE",
            (w // 2 - 80, danger_zone_y + 30),
            self.font_large,
            1.0,
            (0, 0, 255),
            2,
        )
        
        # Warning zone (next 25%)
        warning_zone_y = int(h * 0.50)
        cv2.rectangle(image, (0, warning_zone_y), (w, danger_zone_y), (0, 165, 255), 2)
    
    def _render_terrain_info(self, image: np.ndarray, terrain: Optional[Dict]):
        """Render terrain analysis in corner."""
        if not terrain:
            return
        
        x, y = 20, 100
        
        cv2.putText(
            image,
            "TERRAIN ANALYSIS",
            (x, y),
            self.font_large,
            0.7,
            (255, 255, 255),
            2,
        )
        
        y += 30
        info = [
            f"Formation: {terrain.get('formation', 'Unknown')}",
            f"Soil: {terrain.get('soil_type', 'Unknown')}",
            f"Moisture: {terrain.get('moisture', 0):.0%}",
            f"Hazards: {', '.join(terrain.get('hazards', []))}",
        ]
        
        for line in info:
            cv2.putText(image, line, (x, y), self.font_small, 0.5, (200, 200, 200), 1)
            y += 25
    
    def _render_current_action(self, image: np.ndarray, action: Optional[Dict]):
        """Render current safety action prominently."""
        if not action:
            return
        
        h, w = image.shape[:2]
        action_type = action.get("action_type", "UNKNOWN")
        severity = action.get("severity", 0.0)
        reason = action.get("reason", "")
        
        # Determine color and blinking
        color_map = {
            "EMERGENCY_STOP": (0, 0, 255),
            "URGENT_SLOW": (0, 100, 255),
            "REDUCE_SPEED": (0, 165, 255),
            "MONITOR": (0, 255, 255),
            "CONTINUE": (0, 255, 0),
        }
        color = color_map.get(action_type, (255, 255, 255))
        
        # Large action banner at top
        banner_height = 60
        cv2.rectangle(image, (0, 0), (w, banner_height), color, -1)
        
        cv2.putText(
            image,
            f"ACTION: {action_type}",
            (20, 40),
            self.font_large,
            1.2,
            (0, 0, 0),
            2,
        )
        
        # Reason on second line
        cv2.putText(
            image,
            reason[:80],  # Truncate long reasons
            (20, 70),
            self.font_small,
            0.6,
            (0, 0, 0),
            1,
        )
    
    def _render_system_status(self, image: np.ndarray, status: Optional[Dict]):
        """Render system status in corner."""
        if not status:
            return
        
        h, w = image.shape[:2]
        x, y = w - 300, 20
        
        cv2.putText(
            image,
            "SYSTEM STATUS",
            (x, y),
            self.font_large,
            0.7,
            (255, 255, 255),
            2,
        )
        
        y += 30
        info = [
            f"State: {status.get('system_state', 'UNKNOWN')}",
            f"Detections: {status.get('detections_active', 0)}",
            f"Frame: {status.get('frames_processed', 0)}",
            f"Alerts: {status.get('alerts_issued', 0)}",
            f"E-Stops: {status.get('emergency_stops', 0)}",
        ]
        
        for line in info:
            cv2.putText(image, line, (x, y), self.font_small, 0.5, (200, 200, 200), 1)
            y += 25
    
    def _render_alerts(self, image: np.ndarray):
        """Render active alerts."""
        alerts = self.alert_queue.get_active_alerts()
        
        h, w = image.shape[:2]
        y = h - 30 - (len(alerts) * 35)
        
        for alert in alerts:
            color = alert.get_color()
            
            # Alert box
            cv2.rectangle(image, (10, y), (w - 10, y + 30), color, 2)
            
            # Alert text
            text = f"[{alert.level}] {alert.title}: {alert.message}"
            cv2.putText(
                image,
                text[:100],
                (20, y + 20),
                self.font_small,
                0.6,
                color,
                2,
            )
            
            y += 35
    
    def add_alert(self, level: str, title: str, message: str, duration: float = 5.0):
        """Add alert to queue."""
        alert = VisualAlert(level, title, message, duration)
        self.alert_queue.add_alert(alert)
        logger.info(f"[{level}] {title}: {message}")


class PersistenceRecorder:
    """
    Records monitoring data for offline analysis and audit.
    Persists:
    - All frames with annotations
    - Risk assessments
    - Alerts
    - System decisions
    """
    
    def __init__(self, output_dir: str = "safety_recordings"):
        self.output_dir = output_dir
        self.video_writer = None
        self.frame_log = []
        self.recording_active = False
        
        logger.info(f"Persistence recorder initialized: {output_dir}")
    
    def start_recording(self, output_file: str, fps: int = 30, frame_size: Tuple[int, int] = None):
        """Start recording monitoring video."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.recording_active = True
        
        logger.info(f"Recording started: {output_path}")
    
    def save_frame(self, frame: np.ndarray, metadata: Dict):
        """Save frame with metadata."""
        if self.video_writer:
            self.video_writer.write(frame)
        
        self.frame_log.append({
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        })
    
    def stop_recording(self):
        """Stop recording."""
        if self.video_writer:
            self.video_writer.release()
            self.recording_active = False
            logger.info("Recording stopped")


def test_monitoring_dashboard():
    """Test monitoring dashboard rendering."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger_test = logging.getLogger("dashboard_test")
    
    logger_test.info("=" * 80)
    logger_test.info("MONITORING DASHBOARD TEST")
    logger_test.info("=" * 80)
    
    dashboard = MonitoringDashboard(1920, 1080)
    
    # Create test image
    image = np.full((1080, 1920, 3), [50, 100, 80], dtype=np.uint8)
    
    # Test detections
    detections = {
        0: {
            "bbox": (960, 700, 1040, 900),
            "distance_m": 2.5,
            "risk_level": "WARNING",
            "confidence": 0.95,
        },
        1: {
            "bbox": (500, 600, 580, 850),
            "distance_m": 1.2,
            "risk_level": "HIGH_WARNING",
            "confidence": 0.92,
        },
    }
    
    # Test terrain
    terrain = {
        "formation": "gentle_slope",
        "soil_type": "clay",
        "moisture": 0.65,
        "hazards": ["sticky_mud", "ruts"],
    }
    
    # Test system status
    system_status = {
        "system_state": "MONITORING",
        "detections_active": 2,
        "frames_processed": 150,
        "alerts_issued": 3,
        "emergency_stops": 0,
    }
    
    # Test action
    current_action = {
        "action_type": "REDUCE_SPEED",
        "severity": 0.35,
        "reason": "Person 2.5m ahead on muddy clay - reduce to 60% speed",
    }
    
    # Render
    output = dashboard.render_frame(
        image,
        detections,
        terrain_analysis=terrain,
        system_status=system_status,
        current_action=current_action,
    )
    
    # Add alerts
    dashboard.add_alert("WARNING", "Person Detected", "1 person at 2.5m", duration=3.0)
    dashboard.add_alert("INFO", "Terrain", "Clay soil detected, poor escape routes", duration=5.0)
    
    # Save test image
    import os
    test_dir = os.path.expanduser("~/safety_logs")
    os.makedirs(test_dir, exist_ok=True)
    output_path = os.path.join(test_dir, "dashboard_test.png")
    cv2.imwrite(output_path, output)
    logger_test.info(f"Dashboard test image saved: {output_path}")
    logger_test.info(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_monitoring_dashboard()
