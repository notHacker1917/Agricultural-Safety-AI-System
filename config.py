"""
Centralized Configuration for Agricultural Safety AI System
All thresholds, parameters, and settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class DetectionConfig:
    """Configuration for detection pipeline."""
    # Model settings
    model_path: str = 'yolov8n.pt'
    base_confidence: float = 0.5
    nms_threshold: float = 0.45
    
    # Multi-scale inference
    inference_scales: List[int] = field(default_factory=lambda: [640, 960, 1280])
    scale_merge_iou_threshold: float = 0.5
    
    # SAHI slicing
    sahi_enabled: bool = True
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    
    # Detection filtering
    min_bbox_area_ratio: float = 0.0005  # Minimum 0.05% of frame
    max_bbox_area_ratio: float = 0.5     # Maximum 50% of frame
    min_aspect_ratio: float = 0.2        # Minimum width/height
    max_aspect_ratio: float = 2.0        # Maximum width/height
    edge_margin_ratio: float = 0.02      # Filter detections at edges (2%)
    
    # Temporal stability
    detection_persistence_frames: int = 3  # Require detection in 3 frames
    confidence_smoothing_alpha: float = 0.3  # EMA alpha for confidence
    
    # Agricultural preprocessing
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    brightness_target: int = 127
    shadow_suppression_threshold: int = 30
    dust_noise_kernel_size: int = 3


@dataclass
class TrackingConfig:
    """Configuration for ByteTrack tracker."""
    # Track thresholds
    track_threshold: float = 0.5        # High confidence threshold
    low_match_threshold: float = 0.1    # Low confidence threshold for second matching
    new_track_threshold: float = 0.6    # Threshold for new track confirmation
    
    # Track management
    track_buffer: int = 30              # Frames to keep track before deletion
    match_threshold: float = 0.8        # IoU matching threshold
    min_box_area: float = 100.0         # Minimum bbox area for valid track
    
    # Occlusion handling
    max_occlusion_frames: int = 15      # Max frames for occlusion recovery
    occlusion_confidence_decay: float = 0.9  # Confidence decay during occlusion
    
    # Motion estimation
    velocity_smoothing_alpha: float = 0.2  # EMA alpha for velocity
    max_velocity_pixels_per_frame: float = 50.0  # Maximum expected velocity


@dataclass
class SafetyZoneConfig:
    """Configuration for dynamic safety zones."""
    # Base zone sizes (in meters, will be converted to pixels)
    safe_zone_radius: float = 15.0       # Outer boundary
    warning_zone_radius: float = 10.0    # Warning boundary
    critical_zone_radius: float = 5.0    # Critical boundary
    emergency_zone_radius: float = 2.0   # Emergency boundary
    
    # Dynamic scaling factors
    speed_scaling_factor: float = 0.5    # Zone expansion per m/s of tractor speed
    object_speed_scaling: float = 0.3    # Zone expansion per m/s of object speed
    
    # Direction awareness
    frontal_zone_multiplier: float = 1.5   # Extend zone in front of tractor
    rear_zone_multiplier: float = 1.2      # Extend zone behind tractor
    side_zone_multiplier: float = 1.0      # Normal zone at sides
    
    # Blind spot configuration
    blind_spot_rear_distance: float = 8.0   # Meters behind tractor
    blind_spot_side_angle: float = 30.0     # Degrees from centerline
    blind_spot_risk_multiplier: float = 1.3 # Risk increase in blind spots
    
    # TTC (Time-To-Collision) thresholds
    ttc_critical_threshold: float = 3.0    # Seconds - immediate danger
    ttc_warning_threshold: float = 5.0     # Seconds - warning
    ttc_safe_threshold: float = 10.0       # Seconds - safe
    
    # Risk escalation
    uncertainty_risk_bonus: float = 0.1    # Add risk for uncertain detections
    occlusion_risk_bonus: float = 0.2      # Add risk for occluded objects
    low_confidence_threshold: float = 0.3  # Below this = elevated risk
    low_confidence_risk_bonus: float = 0.15


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Colors (BGR format)
    safe_color: Tuple[int, int, int] = (0, 255, 0)         # Green
    warning_color: Tuple[int, int, int] = (0, 255, 255)    # Yellow
    critical_color: Tuple[int, int, int] = (0, 165, 255)   # Orange
    emergency_color: Tuple[int, int, int] = (0, 0, 255)    # Red
    track_color: Tuple[int, int, int] = (255, 255, 255)    # White
    
    # Zone visualization
    zone_alpha: float = 0.15           # Transparency for zone overlays
    zone_line_thickness: int = 2
    show_zone_labels: bool = True
    
    # Track visualization
    trajectory_length: int = 30        # Frames to show trajectory
    trajectory_thickness: int = 2
    show_track_id: bool = True
    show_confidence: bool = True
    show_velocity: bool = True
    
    # Risk visualization
    risk_label_scale: float = 0.6
    risk_label_thickness: int = 2
    alert_blink_rate: int = 5          # Frames for alert blinking


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Processing
    max_queue_size: int = 10           # Frame processing queue
    async_processing: bool = True
    batch_inference: bool = False      # Enable batch processing
    
    # Memory management
    max_trajectory_history: int = 1000  # Max points per trajectory
    cleanup_interval_frames: int = 100  # Cleanup old data every N frames
    
    # Logging
    log_level: str = 'INFO'
    log_detection_stats: bool = True
    log_performance_metrics: bool = True
    metrics_window_size: int = 30       # Frames for averaging metrics


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and metrics."""
    # Detection evaluation
    enable_evaluation: bool = True
    evaluation_interval: int = 50       # Evaluate every N frames
    
    # Metrics
    precision_threshold: float = 0.5    # IoU threshold for precision
    recall_target: float = 0.9          # Target recall for small objects
    map_evaluation_interval: int = 500  # Evaluate mAP every N frames
    
    # False negative tracking
    track_false_negatives: bool = True
    fn_analysis_window: int = 100       # Frames to analyze for FN


class SystemConfig:
    """Master configuration class."""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.safety = SafetyZoneConfig()
        self.visualization = VisualizationConfig()
        self.performance = PerformanceConfig()
        self.evaluation = EvaluationConfig()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'detection': self.detection.__dict__,
            'tracking': self.tracking.__dict__,
            'safety': self.safety.__dict__,
            'visualization': self.visualization.__dict__,
            'performance': self.performance.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration values."""
        # Detection validation
        assert 0 < self.detection.base_confidence < 1, "Invalid confidence"
        assert all(s > 0 for s in self.detection.inference_scales), "Invalid scales"
        
        # Tracking validation
        assert self.tracking.track_threshold > self.tracking.low_match_threshold, "Invalid thresholds"
        assert self.tracking.track_buffer > 0, "Invalid track buffer"
        
        # Safety validation
        assert self.safety.emergency_zone_radius < self.safety.critical_zone_radius, "Invalid zone sizes"
        assert self.safety.critical_zone_radius < self.safety.warning_zone_radius, "Invalid zone sizes"
        assert self.safety.warning_zone_radius < self.safety.safe_zone_radius, "Invalid zone sizes"
        
        return True


# Global default configuration
DEFAULT_CONFIG = SystemConfig()


def get_config() -> SystemConfig:
    """Get the default system configuration."""
    return DEFAULT_CONFIG