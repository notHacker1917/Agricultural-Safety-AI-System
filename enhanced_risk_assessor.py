#!/usr/bin/env python3
"""
Enhanced 5-Tier Risk Assessment System for Agricultural Safety AI
Provides precise, real-time risk evaluation with complete parameter verification
"""

import numpy as np
import logging
from datetime import datetime

class EnhancedRiskAssessor:
    """
    Comprehensive 5-tier risk assessment with ALL parameter verification.
    
    5-TIER SYSTEM:
    1. SAFE (Green): > 40m away, not in tractor field of view, no collision risk
    2. LOW_WARNING (Light Yellow): 25-40m, somewhat in view, needs situational awareness
    3. WARNING (Dark Yellow): 15-25m, in view, getting closer, active monitoring required
    4. HIGH_WARNING (Orange): 5-15m, directly in danger zone, immediate action needed
    5. CRITICAL (Red): < 5m, imminent collision risk, emergency protocols
    """
    
    def __init__(self, frame_shape=(480, 640), debug=True):
        """
        Initialize enhanced risk assessor.
        
        Args:
            frame_shape: (height, width) of video frame
            debug: Enable detailed logging
        """
        self.height, self.width = frame_shape
        self.debug = debug
        
        # ===== PRECISE DISTANCE THRESHOLDS (meters) =====
        self.dist_critical_max = 5.0       # < 5m: EMERGENCY
        self.dist_high_warn_max = 15.0     # 5-15m: DANGER ZONE
        self.dist_warn_max = 25.0          # 15-25m: CLOSE RANGE
        self.dist_low_warn_max = 40.0      # 25-40m: MEDIUM RANGE
        # > 40m: SAFE
        
        # ===== LATERAL DISTANCE THRESHOLDS (meters) =====
        self.lateral_critical = 3.0        # ±3m: Direct collision zone
        self.lateral_high_warn = 8.0       # ±8m: High danger
        self.lateral_warn = 12.0           # ±12m: Danger zone
        self.lateral_low_warn = 20.0       # ±20m: Medium concern
        # > ±20m: Safe laterally
        
        # ===== TRACTOR PARAMETERS =====
        self.tractor_center_x_norm = 0.5   # Tractor centerline at 50% of frame width
        self.tractor_fov_width = 0.3       # FOV spans 30% width (±15% from center)
        self.tractor_position_y_norm = 0.8  # Tractor at 80% down frame (near camera)
        
        # ===== SPEED FACTORS =====
        self.speed_stationary = 0.0       # No movement
        self.speed_slow = 0.3             # Walking slowly
        self.speed_moderate = 0.6        # Normal walking
        self.speed_fast = 0.9            # Running/fast approach
        
        # ===== MOVEMENT ANALYSIS =====
        self.approaching_risk_mult = 1.8  # 80% risk escalation if approaching
        self.retreating_risk_mult = 0.6   # 40% risk reduction if retreating
        
        logging.info("Enhanced 5-Tier Risk Assessor Initialized")
        self._log_thresholds()
    
    def _log_thresholds(self):
        """Log all thresholds for verification"""
        if self.debug:
            logging.info("=" * 60)
            logging.info("5-TIER RISK ASSESSMENT PARAMETERS")
            logging.info("=" * 60)
            logging.info("DISTANCE ZONES (Forward/Depth):")
            logging.info(f"  CRITICAL (RED):       < {self.dist_critical_max}m (< 5m)")
            logging.info(f"  HIGH_WARNING (ORANGE): {self.dist_critical_max}m - {self.dist_high_warn_max}m")
            logging.info(f"  WARNING (DARK YELLOW): {self.dist_high_warn_max}m - {self.dist_warn_max}m")
            logging.info(f"  LOW_WARNING (LT YELLOW): {self.dist_warn_max}m - {self.dist_low_warn_max}m")
            logging.info(f"  SAFE (GREEN):         > {self.dist_low_warn_max}m")
            logging.info("=" * 60)
            logging.info("LATERAL ZONES (Side-to-Side):")
            logging.info(f"  CRITICAL:       ±{self.lateral_critical}m from centerline")
            logging.info(f"  HIGH_WARNING:   ±{self.lateral_high_warn}m from centerline")
            logging.info(f"  WARNING:        ±{self.lateral_warn}m from centerline")
            logging.info(f"  LOW_WARNING:    ±{self.lateral_low_warn}m from centerline")
            logging.info(f"  SAFE:           > ±{self.lateral_low_warn}m from centerline")
            logging.info("=" * 60)
            logging.info("FIELD OF VIEW (FOV):")
            logging.info(f"  Tractor Center: {self.tractor_center_x_norm*100:.0f}% of frame width")
            logging.info(f"  FOV Width: ±{self.tractor_fov_width*100:.0f}% ({self.tractor_center_x_norm-self.tractor_fov_width/2:.2f} to {self.tractor_center_x_norm+self.tractor_fov_width/2:.2f})")
            logging.info("=" * 60)
    
    def estimate_distance(self, bbox, frame):
        """
        Estimate distance from bounding box position in frame.
        
        PRINCIPLE:
        - Frame Y-axis represents depth: top=far, bottom=close
        - Convert pixel Y-position to estimated distance
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            frame: Video frame (for reference)
            
        Returns:
            float: Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bottom_y = y2
        
        # Y-position in frame (0=top/far, 1=bottom/close)
        y_norm = np.clip(bottom_y / self.height, 0.0, 1.0)
        
        # Linear scale: 0 (top) = 50m, 1 (bottom) = 0m
        estimated_distance = (1.0 - y_norm) * 50.0
        estimated_distance = max(0.1, estimated_distance)  # Minimum 0.1m
        
        return estimated_distance
    
    def estimate_lateral_distance(self, bbox):
        """
        Estimate lateral (side-to-side) distance from center.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            float: Distance from tractor centerline in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_x_norm = bbox_center_x / self.width
        
        # Distance from tractor centerline (normalized)
        centerline_offset_norm = abs(bbox_center_x_norm - self.tractor_center_x_norm)
        
        # Convert to approximate meters (assume 40m field width = frame width)
        field_width_m = 40.0
        lateral_distance = centerline_offset_norm * field_width_m
        
        return lateral_distance
    
    def is_in_fov(self, bbox):
        """
        Check if human is in tractor's field of view.
        
        FOV = ±15% from frame center
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            bool: True if in FOV
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_x_norm = bbox_center_x / self.width
        
        fov_left = self.tractor_center_x_norm - self.tractor_fov_width / 2.0
        fov_right = self.tractor_center_x_norm + self.tractor_fov_width / 2.0
        
        return fov_left <= bbox_center_x_norm <= fov_right
    
    def analyze_movement(self, bbox, prev_bbox, velocity=None):
        """
        Analyze human movement direction and speed.
        
        Args:
            bbox: Current [x1, y1, x2, y2]
            prev_bbox: Previous [x1, y1, x2, y2]
            velocity: Optional pre-computed velocity
            
        Returns:
            dict: Movement analysis with direction and speed
        """
        if prev_bbox is None:
            return {
                'direction': 'unknown',
                'speed_category': 'stationary',
                'speed_norm': 0.0,
                'approaching': False,
                'magnitude': 0.0
            }
        
        # Current and previous center positions
        curr_y = (bbox[1] + bbox[3]) / 2.0
        prev_y = (prev_bbox[1] + prev_bbox[3]) / 2.0
        
        curr_x = (bbox[0] + bbox[2]) / 2.0
        prev_x = (prev_bbox[0] + prev_bbox[2]) / 2.0
        
        # Compute movement
        dy = curr_y - prev_y
        dx = curr_x - prev_x
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Classify direction
        if magnitude < 2:
            direction = 'stationary'
            approaching = False
        elif dy > abs(dx):
            # Moving DOWN (toward camera/tractor = approaching)
            direction = 'down'
            approaching = True
        elif dy < -abs(dx):
            # Moving UP (away from camera/tractor = retreating)
            direction = 'up'
            approaching = False
        elif dx > 0:
            direction = 'right'
            approaching = False
        else:
            direction = 'left'
            approaching = False
        
        # Classify speed
        speed_norm = np.clip(magnitude / 30.0, 0.0, 1.0)  # Normalize by 30 pixels
        
        if magnitude < 2:
            speed_category = 'stationary'
        elif magnitude < 5:
            speed_category = 'slow'
        elif magnitude < 10:
            speed_category = 'moderate'
        elif magnitude < 15:
            speed_category = 'fast'
        else:
            speed_category = 'very_fast'
        
        return {
            'direction': direction,
            'speed_category': speed_category,
            'speed_norm': speed_norm,
            'approaching': approaching,
            'magnitude': magnitude
        }
    
    def compute_risk_level_and_score(self, bbox, prev_bbox=None, velocity=None, frame=None):
        """
        Compute comprehensive 5-tier risk level with ALL parameters verified.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            prev_bbox: Previous frame bbox for movement analysis
            velocity: Pre-computed velocity if available
            frame: Video frame for reference
            
        Returns:
            dict: Complete risk assessment including all parameters
        """
        # Use provided frame or create reference
        if frame is None:
            frame_shape = (self.height, self.width)
        else:
            frame_shape = frame.shape[:2]
        
        # ===== PARAMETER 1: DISTANCE (Forward/Depth) =====
        distance_m = self.estimate_distance(bbox, frame)
        
        # ===== PARAMETER 2: LATERAL DISTANCE (Side-to-Side) =====
        lateral_distance_m = self.estimate_lateral_distance(bbox)
        
        # ===== PARAMETER 3: FIELD OF VIEW =====
        in_fov = self.is_in_fov(bbox)
        
        # ===== PARAMETER 4: MOVEMENT ANALYSIS =====
        movement = self.analyze_movement(bbox, prev_bbox, velocity)
        approaching = movement['approaching']
        speed_norm = movement['speed_norm']
        
        # ===== PARAMETER 5: DIRECTION & SPEED SENTIMENT =====
        direction_sentiment = 1.0
        if approaching:
            direction_sentiment = self.approaching_risk_mult  # Escalate
        elif movement['direction'] == 'up':
            direction_sentiment = self.retreating_risk_mult   # De-escalate
        
        # ===== RISK LEVEL DETERMINATION =====
        # Primary: Check depth zones
        if distance_m <= self.dist_critical_max:
            if lateral_distance_m <= self.lateral_critical:
                base_risk = 'CRITICAL'
                risk_score = 0.95
            elif lateral_distance_m <= self.lateral_high_warn:
                base_risk = 'HIGH_WARNING'
                risk_score = 0.75
            else:
                base_risk = 'WARNING'
                risk_score = 0.55
        
        elif distance_m <= self.dist_high_warn_max:
            if lateral_distance_m <= self.lateral_high_warn and in_fov:
                base_risk = 'HIGH_WARNING'
                risk_score = 0.70
            else:
                base_risk = 'WARNING'
                risk_score = 0.48
        
        elif distance_m <= self.dist_warn_max:
            if lateral_distance_m <= self.lateral_warn and in_fov:
                base_risk = 'WARNING'
                risk_score = 0.45
            else:
                base_risk = 'LOW_WARNING'
                risk_score = 0.28
        
        elif distance_m <= self.dist_low_warn_max:
            if in_fov:
                base_risk = 'LOW_WARNING'
                risk_score = 0.25
            else:
                base_risk = 'SAFE'
                risk_score = 0.10
        
        else:  # > 40m
            base_risk = 'SAFE'
            risk_score = 0.05
        
        # ===== APPLY MOVEMENT MODIFIERS =====
        # Approaching humans: escalate risk
        if approaching and base_risk != 'CRITICAL':
            risk_score = np.clip(risk_score * direction_sentiment, 0.0, 1.0)
            
            # Escalate risk level if approaching
            if base_risk == 'SAFE':
                base_risk = 'LOW_WARNING'
                risk_score = max(risk_score, 0.25)
            elif base_risk == 'LOW_WARNING':
                base_risk = 'WARNING'
                risk_score = max(risk_score, 0.45)
            elif base_risk == 'WARNING':
                base_risk = 'HIGH_WARNING'
                risk_score = max(risk_score, 0.70)
        
        # Retreating humans: de-escalate risk
        elif movement['direction'] == 'up':
            risk_score = np.clip(risk_score * direction_sentiment, 0.0, 1.0)
            
            if base_risk == 'LOW_WARNING':
                base_risk = 'SAFE'
                risk_score = min(risk_score, 0.15)
            elif base_risk == 'WARNING':
                base_risk = 'LOW_WARNING'
                risk_score = min(risk_score, 0.30)
            elif base_risk == 'HIGH_WARNING':
                base_risk = 'WARNING'
                risk_score = min(risk_score, 0.50)
        
        # Speed-based adjustment
        if speed_norm > 0.7:  # Fast movement
            risk_score = np.clip(risk_score * 1.2, 0.0, 1.0)
        
        # Final clipping
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        return {
            # Risk Level
            'risk_level': base_risk,
            'risk_score': risk_score,
            
            # ===== ALL 5 PARAMETERS =====
            # Parameter 1: Distance (Forward)
            'distance_m': distance_m,
            'distance_zone': self._distance_to_zone(distance_m),
            'distance_safe': distance_m > self.dist_low_warn_max,
            
            # Parameter 2: Lateral Distance
            'lateral_distance_m': lateral_distance_m,
            'lateral_zone': self._lateral_to_zone(lateral_distance_m),
            'lateral_safe': lateral_distance_m > self.lateral_low_warn,
            
            # Parameter 3: Field of View
            'in_fov': in_fov,
            'fov_safe': not in_fov,
            'fov_left': self.tractor_center_x_norm - self.tractor_fov_width / 2.0,
            'fov_right': self.tractor_center_x_norm + self.tractor_fov_width / 2.0,
            
            # Parameter 4: Movement Direction
            'direction': movement['direction'],
            'approaching': approaching,
            'direction_safe': not approaching,
            
            # Parameter 5: Speed
            'speed_category': movement['speed_category'],
            'speed_norm': speed_norm,
            'speed_safe': speed_norm < 0.3,
            
            # Summary
            'movement': movement,
            'details': self._format_details(distance_m, lateral_distance_m, in_fov, 
                                           movement['direction'], base_risk, risk_score)
        }
    
    def _distance_to_zone(self, distance_m):
        """Convert distance to zone name"""
        if distance_m < self.dist_critical_max:
            return "CRITICAL_ZONE"
        elif distance_m < self.dist_high_warn_max:
            return "HIGH_WARNING_ZONE"
        elif distance_m < self.dist_warn_max:
            return "WARNING_ZONE"
        elif distance_m < self.dist_low_warn_max:
            return "LOW_WARNING_ZONE"
        else:
            return "SAFE_ZONE"
    
    def _lateral_to_zone(self, lateral_m):
        """Convert lateral distance to zone name"""
        if lateral_m < self.lateral_critical:
            return "CRITICAL_LATERAL"
        elif lateral_m < self.lateral_high_warn:
            return "HIGH_WARNING_LATERAL"
        elif lateral_m < self.lateral_warn:
            return "WARNING_LATERAL"
        elif lateral_m < self.lateral_low_warn:
            return "LOW_WARNING_LATERAL"
        else:
            return "SAFE_LATERAL"
    
    def _format_details(self, distance_m, lateral_m, in_fov, direction,
                       risk_level, risk_score):
        """Format detailed risk assessment string"""
        fov_str = "IN_FOV" if in_fov else "OUT_FOV"
        return (f"5-Tier: {risk_level} (score:{risk_score:.2f}) | "
                f"Dist:{distance_m:.1f}m {self._distance_to_zone(distance_m)} | "
                f"Lateral:{lateral_m:.1f}m | "
                f"FOV:{fov_str} | "
                f"Direction:{direction}")

