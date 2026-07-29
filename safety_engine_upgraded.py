"""
Upgraded Safety Engine for Agricultural Safety AI System
Implements:
- Dynamic safety zones that adapt based on tractor and object state
- Multi-layer zones (SAFE, WARNING, CRITICAL, EMERGENCY)
- Direction-aware risk assessment
- Time-To-Collision (TTC) logic
- Blind spot awareness
- Safety escalation rules
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class SafetyZone:
    """Dynamic safety zone configuration."""
    name: str
    base_radius: float  # Base radius in meters
    color: Tuple[int, int, int]  # BGR color
    risk_level: RiskLevel
    current_radius: float = 0.0  # Dynamically adjusted radius
    
    def __post_init__(self):
        self.current_radius = self.base_radius


@dataclass
class TractorState:
    """Tractor state for dynamic zone calculation."""
    position: Tuple[float, float] = (0.0, 0.0)  # Center position in pixels
    velocity: Tuple[float, float] = (0.0, 0.0)  # Velocity in pixels/frame
    speed: float = 0.0  # Speed magnitude
    heading: float = 0.0  # Heading angle in radians
    is_moving: bool = False
    is_reversing: bool = False
    
    @property
    def speed_ms(self) -> float:
        """Estimate speed in m/s (assuming 30fps and known scale)."""
        # Rough conversion: pixels/frame * 30 frames/s * 0.05 m/pixel
        return self.speed * 30 * 0.05


@dataclass
class ObjectState:
    """Object state for risk calculation."""
    track_id: int
    position: Tuple[float, float]  # Center position
    velocity: Tuple[float, float]  # Velocity
    speed: float  # Speed magnitude
    bbox: Tuple[float, float, float, float]  # Bounding box
    confidence: float  # Detection confidence
    is_occluded: bool = False
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Risk assessment result for an object."""
    track_id: int
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    zone_name: str  # Which zone the object is in
    ttc: Optional[float] = None  # Time-to-collision in seconds
    distance_to_tractor: float = 0.0  # Distance in pixels
    is_in_blind_spot: bool = False
    reason: str = ""
    recommended_action: str = ""


class DynamicSafetyZoneManager:
    """Manages dynamic safety zones around the tractor."""
    
    def __init__(self, config=None):
        """
        Initialize dynamic safety zone manager.
        
        Args:
            config: Configuration object with safety parameters
        """
        # Default zone configurations (radii in meters)
        self.zones = [
            SafetyZone("EMERGENCY", 2.0, (0, 0, 255), RiskLevel.EMERGENCY),    # Red
            SafetyZone("CRITICAL", 5.0, (0, 165, 255), RiskLevel.CRITICAL),    # Orange
            SafetyZone("WARNING", 10.0, (0, 255, 255), RiskLevel.WARNING),     # Yellow
            SafetyZone("SAFE", 15.0, (0, 255, 0), RiskLevel.SAFE),             # Green
        ]
        
        # Dynamic scaling factors
        self.speed_scaling_factor = 0.5      # Zone expansion per m/s
        self.object_speed_scaling = 0.3      # Additional expansion for moving objects
        self.frontal_multiplier = 1.5        # Extend zone in front
        self.rear_multiplier = 1.2           # Extend zone behind
        self.side_multiplier = 1.0           # Normal at sides
        
        # Blind spot configuration
        self.blind_spot_rear_distance = 8.0   # Meters behind tractor
        self.blind_spot_side_angle = 30.0     # Degrees from centerline
        self.blind_spot_risk_multiplier = 1.3
        
        # TTC thresholds (seconds)
        self.ttc_emergency = 3.0
        self.ttc_critical = 5.0
        self.ttc_warning = 10.0
        
        # Risk escalation parameters
        self.uncertainty_bonus = 0.1
        self.occlusion_bonus = 0.2
        self.low_confidence_threshold = 0.3
        self.low_confidence_bonus = 0.15
        
        # Apply config if provided
        if config is not None and hasattr(config, 'safety'):
            self._apply_config(config.safety)
        
        # Tractor geometry (for blind spot calculation)
        self.tractor_width = 2.5  # meters
        self.tractor_length = 4.0  # meters
        
    def _apply_config(self, safety_config):
        """Apply configuration parameters."""
        self.zones[0].base_radius = safety_config.emergency_zone_radius
        self.zones[1].base_radius = safety_config.critical_zone_radius
        self.zones[2].base_radius = safety_config.warning_zone_radius
        self.zones[3].base_radius = safety_config.safe_zone_radius
        
        self.speed_scaling_factor = safety_config.speed_scaling_factor
        self.object_speed_scaling = safety_config.object_speed_scaling
        self.frontal_multiplier = safety_config.frontal_zone_multiplier
        self.rear_multiplier = safety_config.rear_zone_multiplier
        self.side_multiplier = safety_config.side_zone_multiplier
        
        self.blind_spot_rear_distance = safety_config.blind_spot_rear_distance
        self.blind_spot_side_angle = safety_config.blind_spot_side_angle
        self.blind_spot_risk_multiplier = safety_config.blind_spot_risk_multiplier
        
        self.ttc_emergency = safety_config.ttc_critical_threshold
        self.ttc_critical = safety_config.ttc_warning_threshold
        self.ttc_warning = safety_config.ttc_safe_threshold
        
        self.uncertainty_bonus = safety_config.uncertainty_risk_bonus
        self.occlusion_bonus = safety_config.occlusion_risk_bonus
        self.low_confidence_threshold = safety_config.low_confidence_threshold
        self.low_confidence_bonus = safety_config.low_confidence_risk_bonus
    
    def update_zones(self, tractor_state: TractorState, 
                    object_states: List[ObjectState]) -> List[SafetyZone]:
        """
        Dynamically adjust zone sizes based on tractor and object state.
        
        Args:
            tractor_state: Current tractor state
            object_states: List of tracked object states
            
        Returns:
            Updated list of safety zones with adjusted radii
        """
        # Base expansion due to tractor speed
        tractor_speed = tractor_state.speed_ms
        base_expansion = 1.0 + tractor_speed * self.speed_scaling_factor
        
        # Calculate maximum object speed for additional expansion
        max_object_speed = 0.0
        if object_states:
            max_object_speed = max(obj.speed for obj in object_states) * 30 * 0.05  # Convert to m/s estimate
        
        # Additional expansion due to moving objects
        object_expansion = 1.0 + max_object_speed * self.object_speed_scaling
        
        # Combined expansion factor
        total_expansion = base_expansion * object_expansion
        
        # Update zone radii
        for zone in self.zones:
            zone.current_radius = zone.base_radius * total_expansion
        
        return self.zones
    
    def get_zone_at_position(self, distance: float, angle: float, 
                            tractor_heading: float) -> Optional[SafetyZone]:
        """
        Get the safety zone at a given position relative to tractor.
        
        Args:
            distance: Distance from tractor in meters
            angle: Angle from tractor center in radians
            tractor_heading: Tractor heading in radians
            
        Returns:
            SafetyZone if in a zone, None if outside all zones
        """
        # Calculate direction-dependent zone scaling
        relative_angle = angle - tractor_heading
        direction_multiplier = self._get_direction_multiplier(relative_angle)
        
        # Check zones from innermost to outermost
        for zone in sorted(self.zones, key=lambda z: z.current_radius):
            effective_radius = zone.current_radius * direction_multiplier
            if distance <= effective_radius:
                return zone
        
        return None
    
    def _get_direction_multiplier(self, relative_angle: float) -> float:
        """Get zone scaling factor based on direction from tractor."""
        # Normalize angle to [-pi, pi]
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
        
        # Front of tractor (0 radians) gets frontal multiplier
        # Rear of tractor (pi radians) gets rear multiplier
        # Sides get side multiplier
        
        abs_angle = abs(relative_angle)
        
        if abs_angle < np.pi / 4:  # Front 90 degrees
            return self.frontal_multiplier
        elif abs_angle > 3 * np.pi / 4:  # Rear 90 degrees
            return self.rear_multiplier
        else:  # Sides
            return self.side_multiplier
    
    def is_in_blind_spot(self, position: Tuple[float, float], 
                        tractor_state: TractorState) -> bool:
        """
        Check if a position is in the tractor's blind spot.
        
        Args:
            position: Position to check (x, y) in meters relative to tractor
            tractor_state: Current tractor state
            
        Returns:
            True if position is in a blind spot
        """
        x, y = position
        
        # Rear blind spot (directly behind tractor)
        if y < -self.blind_spot_rear_distance:
            # Check if within tractor width
            if abs(x) < self.tractor_width:
                return True
        
        # Side blind spots (at angles)
        angle = np.arctan2(y, x)
        abs_angle = abs(angle)
        
        # Blind spots at approximately 30-60 degrees from centerline
        side_blind_angle_rad = np.radians(self.blind_spot_side_angle)
        if side_blind_angle_rad < abs_angle < np.pi - side_blind_angle_rad:
            distance = np.sqrt(x**2 + y**2)
            if distance < self.blind_spot_rear_distance * 1.5:
                return True
        
        return False
    
    def calculate_ttc(self, object_state: ObjectState, 
                     tractor_state: TractorState) -> Optional[float]:
        """
        Calculate Time-To-Collision (TTC) for an object.
        
        Args:
            object_state: Object state
            tractor_state: Tractor state
            
        Returns:
            TTC in seconds, or None if no collision risk
        """
        # Get relative position and velocity
        rel_pos = (object_state.position[0] - tractor_state.position[0],
                  object_state.position[1] - tractor_state.position[1])
        
        rel_vel = (object_state.velocity[0] - tractor_state.velocity[0],
                  object_state.velocity[1] - tractor_state.velocity[1])
        
        # Distance to tractor
        distance = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
        
        # Relative speed toward tractor
        rel_speed = np.sqrt(rel_vel[0]**2 + rel_vel[1]**2)
        
        if rel_speed < 0.1:  # Not moving significantly
            return None
        
        # Check if object is moving toward tractor
        dot_product = rel_pos[0] * rel_vel[0] + rel_pos[1] * rel_vel[1]
        if dot_product > 0:  # Moving away
            return None
        
        # TTC = distance / closing speed
        ttc = distance / rel_speed
        
        # Convert from frames to seconds (assuming 30fps)
        ttc_seconds = ttc / 30.0
        
        return ttc_seconds
    
    def assess_risk(self, object_state: ObjectState, 
                   tractor_state: TractorState,
                   zones: List[SafetyZone]) -> RiskAssessment:
        """
        Comprehensive risk assessment for an object.
        
        Args:
            object_state: Object state
            tractor_state: Tractor state
            zones: Current safety zones
            
        Returns:
            RiskAssessment with full risk analysis
        """
        # Calculate distance to tractor
        dx = object_state.position[0] - tractor_state.position[0]
        dy = object_state.position[1] - tractor_state.position[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters (rough estimate)
        distance_meters = distance_pixels * 0.05
        
        # Calculate angle from tractor
        angle = np.arctan2(dy, dx)
        
        # Get zone at position
        zone = self.get_zone_at_position(distance_meters, angle, tractor_state.heading)
        
        # Calculate TTC
        ttc = self.calculate_ttc(object_state, tractor_state)
        
        # Check blind spot
        is_blind_spot = self.is_in_blind_spot((dx * 0.05, dy * 0.05), tractor_state)
        
        # Base risk score from zone
        risk_score = 0.0
        if zone:
            zone_risk_map = {
                RiskLevel.EMERGENCY: 1.0,
                RiskLevel.CRITICAL: 0.8,
                RiskLevel.WARNING: 0.5,
                RiskLevel.SAFE: 0.1
            }
            risk_score = zone_risk_map.get(zone.risk_level, 0.0)
        
        # TTC-based risk adjustment
        if ttc is not None:
            if ttc < self.ttc_emergency:
                risk_score = max(risk_score, 1.0)
            elif ttc < self.ttc_critical:
                risk_score = max(risk_score, 0.8)
            elif ttc < self.ttc_warning:
                risk_score = max(risk_score, 0.5)
        
        # Blind spot risk multiplier
        if is_blind_spot:
            risk_score *= self.blind_spot_risk_multiplier
        
        # Occlusion risk bonus
        if object_state.is_occluded:
            risk_score += self.occlusion_bonus
        
        # Low confidence risk bonus
        if object_state.confidence < self.low_confidence_threshold:
            risk_score += self.low_confidence_bonus
        
        # Clamp risk score
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Determine risk level
        if risk_score >= 0.9:
            risk_level = RiskLevel.EMERGENCY
        elif risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.4:
            risk_level = RiskLevel.WARNING
        else:
            risk_level = RiskLevel.SAFE
        
        # Generate reason
        reasons = []
        if zone:
            reasons.append(f"In {zone.name} zone")
        if ttc is not None and ttc < self.ttc_warning:
            reasons.append(f"TTC: {ttc:.1f}s")
        if is_blind_spot:
            reasons.append("In blind spot")
        if object_state.is_occluded:
            reasons.append("Occluded")
        if object_state.confidence < self.low_confidence_threshold:
            reasons.append("Low confidence")
        
        reason = "; ".join(reasons) if reasons else "Outside all zones"
        
        # Generate recommended action
        action_map = {
            RiskLevel.EMERGENCY: "IMMEDIATE STOP - Critical danger!",
            RiskLevel.CRITICAL: "STOP machinery immediately",
            RiskLevel.WARNING: "Slow down and prepare to stop",
            RiskLevel.SAFE: "Continue monitoring"
        }
        recommended_action = action_map.get(risk_level, "Monitor situation")
        
        zone_name = zone.name if zone else "OUTSIDE"
        
        return RiskAssessment(
            track_id=object_state.track_id,
            risk_level=risk_level,
            risk_score=risk_score,
            zone_name=zone_name,
            ttc=ttc,
            distance_to_tractor=distance_pixels,
            is_in_blind_spot=is_blind_spot,
            reason=reason,
            recommended_action=recommended_action
        )


class UpgradedSafetyEngine:
    """
    Upgraded safety engine with dynamic zones and comprehensive risk assessment.
    """
    
    def __init__(self, config=None):
        """
        Initialize upgraded safety engine.
        
        Args:
            config: Configuration object
        """
        self.zone_manager = DynamicSafetyZoneManager(config)
        self.config = config
        
        # State tracking
        self.tractor_state = TractorState()
        self.object_states: Dict[int, ObjectState] = {}
        self.risk_history: Dict[int, List[float]] = {}
        
        # Frame counter
        self.frame_count = 0
        
        # Performance metrics
        self.metrics = {
            'total_assessments': 0,
            'emergency_count': 0,
            'critical_count': 0,
            'warning_count': 0,
            'safe_count': 0
        }
        
        logger.info("Upgraded Safety Engine initialized")
    
    def update_tractor_state(self, position: Tuple[float, float],
                           velocity: Tuple[float, float] = (0, 0),
                           heading: float = 0.0,
                           is_reversing: bool = False):
        """
        Update tractor state.
        
        Args:
            position: Tractor center position (x, y) in pixels
            velocity: Tractor velocity (vx, vy) in pixels/frame
            heading: Tractor heading angle in radians
            is_reversing: Whether tractor is reversing
        """
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        self.tractor_state = TractorState(
            position=position,
            velocity=velocity,
            speed=speed,
            heading=heading,
            is_moving=speed > 0.5,
            is_reversing=is_reversing
        )
    
    def process_frame(self, tracks: List, frame_shape: Tuple) -> List[RiskAssessment]:
        """
        Process a frame and assess risk for all tracked objects.
        
        Args:
            tracks: List of track dictionaries or STrack objects from tracker
            frame_shape: Frame shape (height, width)
            
        Returns:
            List of RiskAssessment for each track
        """
        self.frame_count += 1
        
        # Update object states from tracks
        current_track_ids = set()
        for track in tracks:
            # Handle both dict and STrack objects
            if hasattr(track, 'to_dict'):
                track_dict = track.to_dict()
            elif isinstance(track, dict):
                track_dict = track
            else:
                continue
            
            track_id = track_dict.get('track_id', -1)
            if track_id < 0:
                continue
                
            current_track_ids.add(track_id)
            
            # Create or update object state
            if track_id not in self.object_states:
                self.object_states[track_id] = ObjectState(
                    track_id=track_id,
                    position=track_dict.get('center', track_dict['bbox'][:2]),
                    velocity=track_dict.get('velocity', (0, 0)),
                    speed=np.linalg.norm(track_dict.get('velocity', (0, 0))),
                    bbox=track_dict['bbox'],
                    confidence=track_dict.get('confidence', 0.5),
                    is_occluded=track_dict.get('is_occluded', False)
                )
            else:
                obj = self.object_states[track_id]
                obj.position = track_dict.get('center', track_dict['bbox'][:2])
                obj.velocity = track_dict.get('velocity', (0, 0))
                obj.speed = np.linalg.norm(track_dict.get('velocity', (0, 0)))
                obj.bbox = track_dict['bbox']
                obj.confidence = track_dict.get('confidence', 0.5)
                obj.is_occluded = track_dict.get('is_occluded', False)
            
            # Update trajectory
            obj.trajectory.append(obj.position)
            if len(obj.trajectory) > 30:
                obj.trajectory = obj.trajectory[-30:]
        
        # Remove objects no longer tracked
        ids_to_remove = [tid for tid in self.object_states if tid not in current_track_ids]
        for tid in ids_to_remove:
            del self.object_states[tid]
        
        # Update dynamic zones
        zones = self.zone_manager.update_zones(
            self.tractor_state,
            list(self.object_states.values())
        )
        
        # Assess risk for each object
        assessments = []
        for obj in self.object_states.values():
            assessment = self.zone_manager.assess_risk(obj, self.tractor_state, zones)
            assessments.append(assessment)
            
            # Update risk history
            if obj.track_id not in self.risk_history:
                self.risk_history[obj.track_id] = []
            self.risk_history[obj.track_id].append(assessment.risk_score)
            if len(self.risk_history[obj.track_id]) > 30:
                self.risk_history[obj.track_id] = self.risk_history[obj.track_id][-30:]
            
            # Update metrics
            self.metrics['total_assessments'] += 1
            if assessment.risk_level == RiskLevel.EMERGENCY:
                self.metrics['emergency_count'] += 1
            elif assessment.risk_level == RiskLevel.CRITICAL:
                self.metrics['critical_count'] += 1
            elif assessment.risk_level == RiskLevel.WARNING:
                self.metrics['warning_count'] += 1
            else:
                self.metrics['safe_count'] += 1
        
        return assessments
    
    def get_zone_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate visualization overlays for safety zones.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with zone overlays
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Tractor position (center-bottom by default)
        tractor_x, tractor_y = w // 2, int(h * 0.85)
        
        # Draw zones as ellipses (accounting for perspective)
        for zone in sorted(self.zone_manager.zones, key=lambda z: z.current_radius, reverse=True):
            # Convert meters to pixels (rough estimate)
            radius_pixels = zone.current_radius / 0.05
            
            # Scale for perspective (zones appear smaller at top of frame)
            perspective_scale = 0.5 + 0.5 * (tractor_y / h)
            rx = radius_pixels * perspective_scale
            ry = radius_pixels * perspective_scale * 0.5  # Flatten vertically
            
            # Clamp to reasonable size
            rx = min(rx, w // 2)
            ry = min(ry, h // 3)
            
            # Draw zone ellipse
            cv2.ellipse(overlay, (int(tractor_x), int(tractor_y)), 
                       (int(rx), int(ry)), 0, 0, 360,
                       zone.color, -1)
            
            # Add zone label
            if zone.current_radius > 0:
                label = f"{zone.name} ({zone.current_radius:.1f}m)"
                cv2.putText(overlay, label, 
                           (int(tractor_x) + int(rx) + 10, int(tractor_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 2)
        
        # Blend overlay with original frame
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def get_metrics(self) -> Dict:
        """Get engine metrics."""
        return {
            **self.metrics,
            'frame_count': self.frame_count,
            'active_tracks': len(self.object_states),
            'tractor_speed': self.tractor_state.speed_ms
        }
    
    def reset(self):
        """Reset engine state."""
        self.tractor_state = TractorState()
        self.object_states.clear()
        self.risk_history.clear()
        self.frame_count = 0
        self.metrics = {
            'total_assessments': 0,
            'emergency_count': 0,
            'critical_count': 0,
            'warning_count': 0,
            'safe_count': 0
        }


# Backward compatibility
SafetyEngine = UpgradedSafetyEngine


def test_safety_engine():
    """Test the upgraded safety engine."""
    logger.info("Testing Upgraded Safety Engine")
    
    from config import get_config
    config = get_config()
    
    engine = UpgradedSafetyEngine(config)
    
    # Set tractor state
    engine.update_tractor_state(
        position=(320, 400),
        velocity=(0, -2),
        heading=-np.pi/2,
        is_reversing=False
    )
    
    # Create test tracks
    tracks = [
        {
            'track_id': 0,
            'bbox': (300, 350, 340, 420),
            'center': (320, 385),
            'confidence': 0.9,
            'velocity': (0, 2),
            'is_occluded': False
        },
        {
            'track_id': 1,
            'bbox': (200, 200, 240, 280),
            'center': (220, 240),
            'confidence': 0.8,
            'velocity': (1, 1),
            'is_occluded': False
        }
    ]
    
    # Process frame
    assessments = engine.process_frame(tracks, (480, 640))
    
    for assessment in assessments:
        logger.info(f"Track {assessment.track_id}: {assessment.risk_level.value} "
                   f"(score: {assessment.risk_score:.2f}) - {assessment.reason}")
    
    # Get metrics
    metrics = engine.get_metrics()
    logger.info(f"Metrics: {metrics}")
    
    logger.info("Safety engine test complete")


if __name__ == '__main__':
    test_safety_engine()