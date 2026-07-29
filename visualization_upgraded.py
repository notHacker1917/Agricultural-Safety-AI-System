"""
Upgraded Visualization Module for Agricultural Safety AI System
Implements:
- Dynamic safety zone overlays
- Trajectory prediction visualization
- Risk labels with color coding
- Track confidence display
- Performance metrics overlay
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpgradedVisualizer:
    """Enhanced visualization with dynamic zones and comprehensive overlays."""
    
    def __init__(self, config=None):
        """
        Initialize upgraded visualizer.
        
        Args:
            config: Configuration object with visualization parameters
        """
        # Default colors (BGR format)
        self.colors = {
            'safe': (0, 255, 0),           # Green
            'warning': (0, 255, 255),      # Yellow
            'critical': (0, 165, 255),     # Orange
            'emergency': (0, 0, 255),      # Red
            'track': (255, 255, 255),      # White
            'trajectory': (100, 100, 255), # Light red
            'text': (255, 255, 255),       # White
            'background': (0, 0, 0),       # Black
        }
        
        # Zone colors
        self.zone_colors = {
            'EMERGENCY': (0, 0, 255),
            'CRITICAL': (0, 165, 255),
            'WARNING': (0, 255, 255),
            'SAFE': (0, 255, 0),
            'OUTSIDE': (128, 128, 128)
        }
        
        # Visualization settings
        self.zone_alpha = 0.15
        self.trajectory_thickness = 2
        self.bbox_thickness = 2
        self.text_scale = 0.6
        self.text_thickness = 2
        
        # Apply config if provided
        if config is not None and hasattr(config, 'visualization'):
            self._apply_config(config.visualization)
        
        # Frame counter for animations
        self.frame_count = 0
        
        logger.info("Upgraded Visualizer initialized")
    
    def _apply_config(self, viz_config):
        """Apply configuration parameters."""
        self.colors['safe'] = viz_config.safe_color
        self.colors['warning'] = viz_config.warning_color
        self.colors['critical'] = viz_config.critical_color
        self.colors['emergency'] = viz_config.emergency_color
        self.zone_alpha = viz_config.zone_alpha
        self.trajectory_thickness = viz_config.trajectory_thickness
        self.text_scale = viz_config.risk_label_scale
        self.text_thickness = viz_config.risk_label_thickness
    
    def draw_tracks(self, frame: np.ndarray, tracks: List,
                   risk_assessments: Optional[List[Any]] = None) -> np.ndarray:
        """
        Draw tracked objects with risk-based coloring.
        
        Args:
            frame: Input frame
            tracks: List of track dictionaries or STrack objects
            risk_assessments: Optional list of RiskAssessment objects
            
        Returns:
            Annotated frame
        """
        self.frame_count += 1
        
        # Create risk lookup
        risk_lookup = {}
        if risk_assessments:
            for ra in risk_assessments:
                risk_lookup[ra.track_id] = ra
        
        for track in tracks:
            # Handle both dict and STrack objects
            if hasattr(track, 'to_dict'):
                track_dict = track.to_dict()
            elif isinstance(track, dict):
                track_dict = track
            else:
                continue
            
            track_id = track_dict.get('track_id', -1)
            bbox = track_dict.get('bbox', (0, 0, 0, 0))
            confidence = track_dict.get('confidence', 0.5)
            is_occluded = track_dict.get('is_occluded', False)
            trajectory = track_dict.get('trajectory', [])
            
            # Get risk assessment
            risk = risk_lookup.get(track_id)
            
            # Determine color based on risk
            if is_occluded:
                color = (0, 0, 255)  # Red for occluded
            elif risk:
                color = self._get_risk_color(risk.risk_score)
            else:
                color = self.colors['track']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # Draw trajectory
            if len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, self.trajectory_thickness)
            
            # Draw labels
            labels = []
            
            # Track ID
            labels.append(f"ID: {track_id}")
            
            # Confidence
            labels.append(f"Conf: {confidence:.2f}")
            
            # Risk level
            if risk:
                labels.append(f"{risk.risk_level.value}")
                if risk.ttc is not None:
                    labels.append(f"TTC: {risk.ttc:.1f}s")
            
            # Occlusion status
            if is_occluded:
                labels.append("OCCLUDED")
            
            # Draw labels
            y_offset = y1 - 10
            for i, label in enumerate(labels):
                y_pos = y_offset - i * 20
                cv2.putText(frame, label, (x1, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, self.text_scale * 0.8,
                           color, self.text_thickness)
        
        return frame
    
    def draw_safety_zones(self, frame: np.ndarray, zones: List[Any],
                         tractor_position: Tuple[float, float],
                         heading: float = 0.0) -> np.ndarray:
        """
        Draw dynamic safety zones around tractor.
        
        Args:
            frame: Input frame
            zones: List of SafetyZone objects
            tractor_position: Tractor center position (x, y)
            heading: Tractor heading angle in radians
            
        Returns:
            Frame with zone overlays
        """
        h, w = frame.shape[:2]
        tractor_x, tractor_y = int(tractor_position[0]), int(tractor_position[1])
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw zones from largest to smallest
        for zone in sorted(zones, key=lambda z: z.current_radius, reverse=True):
            # Convert meters to pixels
            radius_pixels = zone.current_radius / 0.05  # Rough conversion
            
            # Apply perspective scaling
            perspective_scale = 0.5 + 0.5 * (tractor_y / h)
            rx = int(radius_pixels * perspective_scale)
            ry = int(radius_pixels * perspective_scale * 0.4)
            
            # Clamp to reasonable size
            rx = min(rx, w // 2)
            ry = min(ry, h // 3)
            
            # Draw zone as filled ellipse
            cv2.ellipse(overlay, (tractor_x, tractor_y),
                       (rx, ry), np.degrees(heading), 0, 360,
                       zone.color, -1)
            
            # Draw zone boundary
            cv2.ellipse(frame, (tractor_x, tractor_y),
                       (rx, ry), np.degrees(heading), 0, 360,
                       zone.color, 2)
            
            # Draw zone label
            if zone.current_radius > 0:
                label_x = tractor_x + rx + 10
                label_y = tractor_y - ry // 2
                label = f"{zone.name}: {zone.current_radius:.1f}m"
                cv2.putText(frame, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 2)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.zone_alpha, frame, 1 - self.zone_alpha, 0, frame)
        
        # Draw tractor marker
        cv2.circle(frame, (tractor_x, tractor_y), 8, (255, 0, 255), -1)
        cv2.circle(frame, (tractor_x, tractor_y), 12, (255, 0, 255), 2)
        
        # Draw heading indicator
        heading_length = 30
        end_x = int(tractor_x + heading_length * np.cos(heading - np.pi/2))
        end_y = int(tractor_y + heading_length * np.sin(heading - np.pi/2))
        cv2.arrowedLine(frame, (tractor_x, tractor_y), (end_x, end_y),
                       (255, 0, 255), 2)
        
        return frame
    
    def draw_system_status(self, frame: np.ndarray, 
                          detection_metrics: Optional[Dict] = None,
                          tracking_stats: Optional[Dict] = None,
                          safety_metrics: Optional[Dict] = None,
                          fps: float = 0.0) -> np.ndarray:
        """
        Draw system status overlay.
        
        Args:
            frame: Input frame
            detection_metrics: Detection performance metrics
            tracking_stats: Tracking statistics
            safety_metrics: Safety engine metrics
            fps: Current FPS
            
        Returns:
            Frame with status overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        panel_width = 280
        panel_height = 200
        
        # Adjust height based on available info
        info_lines = 10
        if detection_metrics:
            info_lines += 3
        if tracking_stats:
            info_lines += 3
        if safety_metrics:
            info_lines += 4
        
        panel_height = info_lines * 22 + 20
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 1 - 0.7, 0, frame)
        
        # Draw status information
        y = 20
        
        # FPS
        fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y += 22
        
        # Frame count
        if safety_metrics:
            cv2.putText(frame, f"Frame: {safety_metrics.get('frame_count', 0)}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y += 22
        
        # Active tracks
        if tracking_stats:
            active = tracking_stats.get('active_tracks', 0)
            cv2.putText(frame, f"Tracks: {active}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y += 22
        
        # Risk summary
        if safety_metrics:
            y += 5
            cv2.putText(frame, "--- Risk Summary ---",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 22
            
            emergency = safety_metrics.get('emergency_count', 0)
            critical = safety_metrics.get('critical_count', 0)
            warning = safety_metrics.get('warning_count', 0)
            safe = safety_metrics.get('safe_count', 0)
            
            if emergency > 0:
                cv2.putText(frame, f"EMERGENCY: {emergency}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 22
            if critical > 0:
                cv2.putText(frame, f"CRITICAL: {critical}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                y += 22
            
            cv2.putText(frame, f"WARNING: {warning}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 22
            cv2.putText(frame, f"SAFE: {safe}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 22
        
        # Detection metrics
        if detection_metrics:
            y += 5
            cv2.putText(frame, "--- Detection ---",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 22
            
            total = detection_metrics.get('total_detections', 0)
            small = detection_metrics.get('small_object_detections', 0)
            precision = detection_metrics.get('precision_estimate', 0)
            
            cv2.putText(frame, f"Detections: {total}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y += 22
            cv2.putText(frame, f"Small objects: {small}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y += 22
            cv2.putText(frame, f"Precision: {precision:.2f}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return frame
    
    def draw_alert_banner(self, frame: np.ndarray, alert_level: str,
                         message: str) -> np.ndarray:
        """
        Draw alert banner for critical situations.
        
        Args:
            frame: Input frame
            alert_level: Alert level (EMERGENCY, CRITICAL, WARNING)
            message: Alert message
            
        Returns:
            Frame with alert banner
        """
        h, w = frame.shape[:2]
        
        # Determine colors based on alert level
        if alert_level == 'EMERGENCY':
            bg_color = (0, 0, 255)
            text_color = (255, 255, 255)
            blink = self.frame_count % 10 < 5  # Blink effect
        elif alert_level == 'CRITICAL':
            bg_color = (0, 165, 255)
            text_color = (0, 0, 0)
            blink = True
        else:  # WARNING
            bg_color = (0, 255, 255)
            text_color = (0, 0, 0)
            blink = False
        
        if not blink:
            return frame
        
        # Draw banner background
        banner_height = 60
        cv2.rectangle(frame, (0, h // 2 - banner_height // 2),
                     (w, h // 2 + banner_height // 2), bg_color, -1)
        
        # Draw alert text
        cv2.putText(frame, f"⚠ {alert_level} ⚠",
                   (w // 2 - 150, h // 2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        
        cv2.putText(frame, message,
                   (w // 2 - 200, h // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return frame
    
    def _get_risk_color(self, risk_score: float) -> Tuple[int, int, int]:
        """Get color based on risk score."""
        if risk_score >= 0.9:
            return self.colors['emergency']
        elif risk_score >= 0.7:
            return self.colors['critical']
        elif risk_score >= 0.4:
            return self.colors['warning']
        else:
            return self.colors['safe']
    
    def draw_prediction_trajectories(self, frame: np.ndarray,
                                    predictions: List[Dict]) -> np.ndarray:
        """
        Draw predicted future trajectories.
        
        Args:
            frame: Input frame
            predictions: List of prediction dictionaries with 'track_id' and 'predicted_path'
            
        Returns:
            Frame with prediction visualizations
        """
        for pred in predictions:
            track_id = pred.get('track_id')
            predicted_path = pred.get('trajectory', [])
            
            if not predicted_path:
                continue
            
            # Get color based on track risk (default to blue for predictions)
            color = (255, 0, 0)  # Blue for predictions
            
            # Draw predicted path as dotted line
            for i in range(len(predicted_path) - 1):
                pt1 = (int(predicted_path[i][0]), int(predicted_path[i][1]))
                pt2 = (int(predicted_path[i+1][0]), int(predicted_path[i+1][1]))
                
                # Dotted effect
                if i % 2 == 0:
                    cv2.line(frame, pt1, pt2, color, 2)
                else:
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
            
            # Draw prediction endpoint
            if predicted_path:
                end_point = (int(predicted_path[-1][0]), int(predicted_path[-1][1]))
                cv2.circle(frame, end_point, 5, color, -1)
        
        return frame
    
    def reset(self):
        """Reset visualizer state."""
        self.frame_count = 0


# Backward compatibility
Visualizer = UpgradedVisualizer


def test_visualizer():
    """Test the upgraded visualizer."""
    logger.info("Testing Upgraded Visualizer")
    
    from config import get_config
    config = get_config()
    
    visualizer = UpgradedVisualizer(config)
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test tracks
    tracks = [
        {
            'track_id': 0,
            'bbox': (300, 350, 340, 420),
            'center': (320, 385),
            'confidence': 0.9,
            'velocity': (0, 2),
            'trajectory': [(320, 385), (320, 380), (320, 375)],
            'is_occluded': False
        }
    ]
    
    # Draw tracks
    frame = visualizer.draw_tracks(test_frame.copy(), tracks)
    
    # Test system status
    status_info = {
        'frame_count': 100,
        'total_assessments': 150,
        'emergency_count': 0,
        'critical_count': 2,
        'warning_count': 10,
        'safe_count': 138
    }
    
    frame = visualizer.draw_system_status(frame, safety_metrics=status_info, fps=25.0)
    
    logger.info(f"Test complete. Frame shape: {frame.shape}")


if __name__ == '__main__':
    test_visualizer()