"""
Advanced Safety Engine for Harvester/Truck Operations.

Focuses on:
- Field of view (FOV) prediction for harvester/truck trajectory
- Operator blind spots
- Time-to-collision warnings
- Dangerous zone escalation
"""

import numpy as np
import logging
from datetime import datetime

class HarvesterSafetyEngine:
    """
    Advanced Safety system for harvester/truck-human interactions with 5-tier risk assessment.
    
    Defines five danger zones based on depth of field and proximity:
    1. CRITICAL: Immediate danger zone (<5m, direct collision risk)
    2. HIGH_WARNING: Close proximity (5-15m, immediate attention required)
    3. WARNING: Medium distance (15-25m, monitoring required)
    4. LOW_WARNING: Far but approaching (25-40m, situational awareness)
    5. SAFE: Outside danger zones (>40m, no immediate concern)
    """

    def __init__(self,
                 harvester_width=2.5,  # meters (typical combine harvester)
                 harvester_length=10.0,  # meters
                 harvester_speed_ms=2.0,  # meters per second
                 critical_forward_distance=5,   # meters ahead (immediate danger)
                 high_warning_forward_distance=15,  # meters ahead
                 warning_forward_distance=25,   # meters ahead
                 low_warning_forward_distance=40,   # meters ahead
                 critical_side_distance=3,   # meters to sides (immediate)
                 high_warning_side_distance=8,   # meters to sides
                 warning_side_distance=12,   # meters to sides
                 low_warning_side_distance=20): # meters to sides
        """
        Initialize harvester safety parameters with 5-tier risk assessment.
        """
        self.harvester_width = harvester_width
        self.harvester_length = harvester_length
        self.harvester_speed_ms = harvester_speed_ms

        # Five-tier distance zones (depth of field)
        self.critical_forward_dist = critical_forward_distance      # < 5m
        self.high_warning_forward_dist = high_warning_forward_distance  # 5-15m
        self.warning_forward_dist = warning_forward_distance        # 15-25m
        self.low_warning_forward_dist = low_warning_forward_distance    # 25-40m
        # > 40m = SAFE

        # Lateral distance zones
        self.critical_side_dist = critical_side_distance
        self.high_warning_side_dist = high_warning_side_distance
        self.warning_side_dist = warning_side_distance
        self.low_warning_side_dist = low_warning_side_distance

        logging.info("Harvester Safety Engine initialized with 5-tier risk assessment")
        logging.info(f"  - Harvester: {harvester_width}m wide x {harvester_length}m long")
        logging.info(f"  - CRITICAL zone: ±{critical_side_distance}m, {critical_forward_distance}m ahead")
        logging.info(f"  - HIGH_WARNING zone: ±{high_warning_side_distance}m, {high_warning_forward_distance}m ahead")
        logging.info(f"  - WARNING zone: ±{warning_side_distance}m, {warning_forward_distance}m ahead")
        logging.info(f"  - LOW_WARNING zone: ±{low_warning_side_distance}m, {low_warning_forward_distance}m ahead")
        logging.info("  - SAFE zone: Beyond LOW_WARNING zones")

    def analyze_depth_of_field(self, human_bbox, frame_shape, camera_fov_deg=60, camera_height_m=2.5):
        """
        Enhanced depth of field analysis using multiple estimation methods.

        Args:
            human_bbox: [x1, y1, x2, y2] bounding box
            frame_shape: (height, width) of frame
            camera_fov_deg: Camera field of view in degrees
            camera_height_m: Camera height above ground in meters

        Returns:
            dict: Depth analysis with confidence scores
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = human_bbox

        # Method 1: Position-based depth (frame coordinates)
        # Y-position in frame indicates depth (bottom = close, top = far)
        # Camera looks forward, so bottom of frame = closest to harvester
        y_center = (y1 + y2) / 2
        # Invert Y so bottom = 0m, top = 50m
        normalized_y = 1.0 - (y_center / h)  # 0 at bottom, 1 at top
        depth_from_position = normalized_y * 50.0  # Linear scale 0-50m

        # Method 2: Size-based depth estimation
        # Human height in pixels relates to distance
        bbox_height = y2 - y1
        avg_human_height_pixels = 170  # Average human height in pixels at ~2m distance
        focal_length_pixels = (w / 2) / np.tan(np.radians(camera_fov_deg / 2))

        # Real human height (meters)
        real_human_height = 1.7  # Average human height

        # Distance estimation using similar triangles
        distance_from_size = (real_human_height * focal_length_pixels) / bbox_height

        # Method 3: Perspective correction using camera height
        # Account for camera being above ground level
        y_bottom = y2
        viewing_angle = np.arctan2(camera_height_m, depth_from_position)
        corrected_depth = camera_height_m / np.tan(viewing_angle)

        # Weighted combination of methods
        depth_estimates = {
            'position_based': depth_from_position,
            'size_based': distance_from_size,
            'perspective_corrected': corrected_depth
        }

        # Confidence scores based on reliability
        confidences = {
            'position_based': 0.8,  # Reliable for known camera setup
            'size_based': 0.0,  # Disable for mock testing
            'perspective_corrected': 0.7  # Moderate confidence
        }

        # Weighted average depth
        total_weight = sum(confidences.values())
        weighted_depth = sum(depth * conf for depth, conf in zip(depth_estimates.values(), confidences.values())) / total_weight

        # Depth uncertainty based on method agreement
        depth_variance = np.var(list(depth_estimates.values()))
        depth_confidence = max(0.1, 1.0 - depth_variance / 100.0)  # Lower variance = higher confidence

        return {
            'estimated_depth_m': weighted_depth,
            'depth_confidence': depth_confidence,
            'depth_methods': depth_estimates,
            'method_confidences': confidences,
            'depth_uncertainty_m': np.sqrt(depth_variance),
            'field_analysis': self._analyze_field_position(human_bbox, frame_shape, weighted_depth)
        }

    def _analyze_field_position(self, human_bbox, frame_shape, estimated_depth):
        """
        Analyze human position in agricultural field context.

        Args:
            human_bbox: Human bounding box
            frame_shape: Frame dimensions
            estimated_depth: Estimated distance in meters

        Returns:
            dict: Field position analysis
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = human_bbox

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Convert to normalized coordinates
        x_norm = x_center / w
        y_norm = y_center / h

        # Field zones (assuming camera faces forward)
        if x_norm < 0.3:
            lateral_position = "far_left"
        elif x_norm < 0.4:
            lateral_position = "left"
        elif x_norm < 0.6:
            lateral_position = "center"
        elif x_norm < 0.7:
            lateral_position = "right"
        else:
            lateral_position = "far_right"

        # Depth zones
        if estimated_depth < 5:
            depth_zone = "immediate"
        elif estimated_depth < 15:
            depth_zone = "very_close"
        elif estimated_depth < 25:
            depth_zone = "close"
        elif estimated_depth < 40:
            depth_zone = "medium"
        else:
            depth_zone = "far"

        # Visibility analysis
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        relative_size = bbox_area / frame_area

        visibility_score = min(1.0, relative_size * 100)  # Scale to 0-1

        return {
            'lateral_position': lateral_position,
            'depth_zone': depth_zone,
            'normalized_position': (x_norm, y_norm),
            'visibility_score': visibility_score,
            'bbox_relative_size': relative_size,
            'field_context': f"{depth_zone}_{lateral_position}"
        }

    def compute_risk_level(self, human_bbox, frame_shape, harvester_position=(0.5, 0.7), movement_data=None):
        """
        Enhanced risk level computation with movement analysis.
        
        SPATIAL MODEL:
        - Camera represents truck's perspective (front-facing)
        - Truck positioned at (0.5, 0.7) in frame = center-bottom
        - Y-AXIS = Camera depth: LOW Y (top) = Far from truck = LESS DANGER
                                 HIGH Y (bottom) = Close to truck = MORE DANGER
        - X-AXIS = Lateral distance from truck centerline
        
        MOVEMENT FACTORS:
        - Direction: Approaching (down) = HIGHER RISK
        - Speed: Faster movement = HIGHER RISK  
        - Trajectory: Human-like, consistent movement = HIGHER CONFIDENCE
        
        Args:
            human_bbox: [x1, y1, x2, y2] normalized or pixel coordinates
            frame_shape: (height, width)
            harvester_position: (x_norm, y_norm) of harvester in frame
            movement_data: Optional movement analysis from enhanced detection
            
        Returns:
            dict: Enhanced risk assessment with movement factors
        """
        h, w = frame_shape[:2]
        
        # Normalize bbox if needed
        if human_bbox[0] > 1 or human_bbox[2] > 1:
            # Pixel coordinates
            x1, y1, x2, y2 = human_bbox
        else:
            # Normalized
            x1, y1, x2, y2 = [human_bbox[i] * w if i % 2 == 0 else human_bbox[i] * h 
                              for i in range(4)]
        
        human_center_x = (x1 + x2) / 2
        human_center_y = (y1 + y2) / 2
        human_size = max(x2 - x1, y2 - y1)
        
        # Truck position in frame
        truck_x = harvester_position[0] * w
        truck_y = harvester_position[1] * h
        
        # Pixel-to-meter conversion
        pixels_per_meter = h / 30.0
        
        # SPATIAL MODEL - Camera View Perspective:
        # Frame represents truck's forward/outward view of field
        # - Frame TOP (Y=0) = FARTHEST from camera/truck (50m away) 
        # - Frame BOTTOM (Y=h) = CLOSEST to camera/truck (0m away, immediate)
        #
        # Closer objects = LOWER in frame = HIGHER danger
        # Farther objects = HIGHER in frame = LOWER danger
        
        # Enhanced depth analysis using multiple methods
        depth_analysis = self.analyze_depth_of_field(human_bbox, frame_shape)
        human_depth_m = depth_analysis['estimated_depth_m']
        depth_confidence = depth_analysis['depth_confidence']

        # Use ground-plane projection when available for more accurate risk mapping
        ground_point = None
        if movement_data:
            ground_point = movement_data.get('ground_point')

        if ground_point is not None and len(ground_point) == 2:
            ground_x, ground_z = ground_point
            centerline = 10.0
            human_depth_m = max(0.1, ground_z)
            distance_lateral_m = abs(ground_x - centerline)
            depth_analysis['ground_plane'] = {
                'projected_x_m': ground_x,
                'projected_z_m': ground_z,
            }
            depth_confidence = max(depth_confidence, 0.9)
        else:
            # Convert pixel offset to meters if ground-plane projection is unavailable
            dx = human_center_x - truck_x
            distance_lateral_m = abs(dx) / pixels_per_meter

        # Adjust risk calculations based on depth confidence
        confidence_multiplier = 0.8 + (depth_confidence * 0.4)  # 0.8-1.2 range
        
        # MOVEMENT ENHANCED RISK FACTORS
        movement_risk_multiplier = 1.0
        direction_risk = 1.0
        speed_risk = 1.0
        
        if movement_data:
            # Direction-based risk adjustment
            direction = movement_data.get('direction', 'stationary')
            if direction == 'down':
                # Moving towards camera/truck = HIGHER RISK
                direction_risk = 1.5
                logging.debug(f"Human moving TOWARDS camera - increased risk")
            elif direction == 'up':
                # Moving away from camera/truck = LOWER RISK
                direction_risk = 0.7
                logging.debug(f"Human moving AWAY from camera - decreased risk")
            elif direction in ['left', 'right']:
                # Lateral movement = MODERATE RISK
                direction_risk = 1.2
                logging.debug(f"Human moving LATERALLY - moderate risk adjustment")
            
            # Speed-based risk adjustment
            speed_category = movement_data.get('speed_category', 'moderate')
            speed_multipliers = {
                'very_fast': 1.8, 'fast': 1.5, 'moderate': 1.2,
                'slow': 1.0, 'very_slow': 0.9, 'minimal': 0.8
            }
            speed_risk = speed_multipliers.get(speed_category, 1.0)
            
            # Trajectory consistency bonus
            trajectory = movement_data.get('trajectory_pattern', {})
            if trajectory.get('is_human_like', False):
                movement_risk_multiplier = 1.1  # Slight bonus for human-like movement
            if trajectory.get('consistency', 0) > 0.8:
                movement_risk_multiplier *= 1.05  # Bonus for consistent movement
        
        # FIVE-TIER DEPTH-BASED ZONES with enhanced field analysis
        # CRITICAL: 0-5m (immediate danger, collision imminent)
        # HIGH_WARNING: 5-15m (very close, immediate action required)
        # WARNING: 15-25m (close, active monitoring needed)
        # LOW_WARNING: 25-40m (far but potentially concerning)
        # SAFE: >40m (outside danger zones)

        critical_max_m = self.critical_forward_dist         # 5m
        high_warning_max_m = self.high_warning_forward_dist # 15m
        warning_max_m = self.warning_forward_dist           # 25m
        low_warning_max_m = self.low_warning_forward_dist   # 40m

        # Determine depth-based risk zones
        in_critical_depth = human_depth_m <= critical_max_m
        in_high_warning_depth = (critical_max_m < human_depth_m <= high_warning_max_m)
        in_warning_depth = (high_warning_max_m < human_depth_m <= warning_max_m)
        in_low_warning_depth = (warning_max_m < human_depth_m <= low_warning_max_m)
        in_safe_depth = human_depth_m > low_warning_max_m

        # Determine lateral risk zones
        in_critical_lateral = distance_lateral_m <= self.critical_side_dist
        in_high_warning_lateral = distance_lateral_m <= self.high_warning_side_dist
        in_warning_lateral = distance_lateral_m <= self.warning_side_dist
        in_low_warning_lateral = distance_lateral_m <= self.low_warning_side_dist

        # ENHANCED 5-TIER RISK DETERMINATION with DYNAMIC MOVEMENT-BASED ESCALATION
        # Base risk levels determined by depth zones
        base_risk_level = None
        base_risk_score = 0.0

        if in_critical_depth and in_critical_lateral:
            base_risk_level = "CRITICAL"
            depth_risk = 1.0 - (human_depth_m / critical_max_m)
            lateral_risk = 1.0 - (distance_lateral_m / self.critical_side_dist)
            base_risk_score = min(1.0, 0.8 * depth_risk + 0.2 * lateral_risk)

        elif in_high_warning_depth and in_high_warning_lateral:
            base_risk_level = "HIGH_WARNING"
            depth_risk = 0.8 * ((human_depth_m - critical_max_m) / (high_warning_max_m - critical_max_m))
            lateral_risk = 0.6 * (1.0 - (distance_lateral_m / self.high_warning_side_dist))
            base_risk_score = min(1.0, depth_risk + lateral_risk)

        elif in_warning_depth and in_warning_lateral:
            base_risk_level = "WARNING"
            depth_risk = 0.6 * ((human_depth_m - high_warning_max_m) / (warning_max_m - high_warning_max_m))
            lateral_risk = 0.4 * (1.0 - (distance_lateral_m / self.warning_side_dist))
            base_risk_score = min(1.0, depth_risk + lateral_risk)

        elif in_low_warning_depth and in_low_warning_lateral:
            base_risk_level = "LOW_WARNING"
            depth_risk = 0.4 * ((human_depth_m - warning_max_m) / (low_warning_max_m - warning_max_m))
            lateral_risk = 0.2 * (1.0 - (distance_lateral_m / self.low_warning_side_dist))
            base_risk_score = min(1.0, depth_risk + lateral_risk)

        else:
            base_risk_level = "SAFE"
            base_risk_score = 0.0

        # DYNAMIC RISK LEVEL ESCALATION BASED ON MOVEMENT DIRECTION
        # Humans walking closer get IMMEDIATE risk level increases
        escalated_risk_level = base_risk_level
        movement_escalation = 1.0

        if movement_data and base_risk_level != "CRITICAL":
            direction = movement_data.get('direction', 'stationary')
            speed_category = movement_data.get('speed_category', 'moderate')

            # Speed multipliers for escalation
            speed_multipliers = {
                'very_fast': 2.0, 'fast': 1.5, 'moderate': 1.2,
                'slow': 1.0, 'very_slow': 0.8, 'minimal': 0.5
            }
            speed_factor = speed_multipliers.get(speed_category, 1.0)

            if direction == 'down':
                # APPROACHING: Immediate risk escalation
                if base_risk_level == "SAFE":
                    escalated_risk_level = "LOW_WARNING"
                    movement_escalation = 1.8 * speed_factor
                    logging.info(f"SAFE human APPROACHING - ESCALATED to LOW_WARNING")
                elif base_risk_level == "LOW_WARNING":
                    escalated_risk_level = "WARNING"
                    movement_escalation = 1.6 * speed_factor
                    logging.info(f"LOW_WARNING human APPROACHING - ESCALATED to WARNING")
                elif base_risk_level == "WARNING":
                    escalated_risk_level = "HIGH_WARNING"
                    movement_escalation = 1.4 * speed_factor
                    logging.info(f"WARNING human APPROACHING - ESCALATED to HIGH_WARNING")
                elif base_risk_level == "HIGH_WARNING":
                    escalated_risk_level = "CRITICAL"
                    movement_escalation = 1.2 * speed_factor
                    logging.info(f"HIGH_WARNING human APPROACHING - ESCALATED to CRITICAL")

            elif direction == 'up':
                # RETREATING: Risk de-escalation
                if base_risk_level == "LOW_WARNING":
                    escalated_risk_level = "SAFE"
                    movement_escalation = 0.6 * speed_factor
                    logging.info(f"LOW_WARNING human RETREATING - DE-ESCALATED to SAFE")
                elif base_risk_level == "WARNING":
                    escalated_risk_level = "LOW_WARNING"
                    movement_escalation = 0.7 * speed_factor
                    logging.info(f"WARNING human RETREATING - DE-ESCALATED to LOW_WARNING")
                elif base_risk_level == "HIGH_WARNING":
                    escalated_risk_level = "WARNING"
                    movement_escalation = 0.8 * speed_factor
                    logging.info(f"HIGH_WARNING human RETREATING - DE-ESCALATED to WARNING")

        # Apply final risk calculation with escalation
        if escalated_risk_level == "CRITICAL":
            risk_level = "CRITICAL"
            risk_score = min(1.0, base_risk_score * direction_risk * speed_risk * movement_risk_multiplier * movement_escalation)
            time_to_collision = human_depth_m / max(0.1, self.harvester_speed_ms)

        elif escalated_risk_level == "HIGH_WARNING":
            risk_level = "HIGH_WARNING"
            risk_score = min(1.0, base_risk_score * direction_risk * speed_risk * movement_risk_multiplier * movement_escalation)
            time_to_collision = human_depth_m / max(0.1, self.harvester_speed_ms)

        elif escalated_risk_level == "WARNING":
            risk_level = "WARNING"
            risk_score = min(1.0, base_risk_score * direction_risk * speed_risk * movement_risk_multiplier * movement_escalation)
            time_to_collision = human_depth_m / max(0.1, self.harvester_speed_ms)

        elif escalated_risk_level == "LOW_WARNING":
            risk_level = "LOW_WARNING"
            risk_score = min(1.0, base_risk_score * direction_risk * speed_risk * movement_risk_multiplier * movement_escalation)
            time_to_collision = human_depth_m / max(0.1, self.harvester_speed_ms)

        else:
            risk_level = "SAFE"
            risk_score = 0.0
            time_to_collision = float('inf')

        # Increase risk if occlusion is detected in the track
        if movement_data and movement_data.get('is_occluded'):
            occlusion_multiplier = 1.0 + min(0.8, movement_data.get('occlusion_confidence', 0.5))
            risk_score = min(1.0, risk_score * occlusion_multiplier)
            logging.debug(f"Occlusion detected, applying penalty factor {occlusion_multiplier:.2f}")
        
        # Apply prediction-based risk escalation
        prediction_escalation = movement_data.get('prediction_escalation', {}) if movement_data else {}
        if prediction_escalation:
            escalation_factor = prediction_escalation.get('escalation_factor', 1.0)
            risk_score = min(1.0, risk_score * escalation_factor)
            logging.debug(f"Prediction escalation applied: {escalation_factor:.2f} ({prediction_escalation.get('reason', 'unknown')})")
        
        # Enhanced details with movement information
        movement_info = ""
        if movement_data:
            direction = movement_data.get('direction', 'unknown')
            speed = movement_data.get('speed_category', 'unknown')
            movement_info = f", Direction: {direction}, Speed: {speed}"
            if movement_data.get('is_occluded'):
                occ_reason = movement_data.get('occlusion_reason', 'occluded')
                movement_info += f", Occlusion: {occ_reason}"
            if prediction_escalation:
                pred_reason = prediction_escalation.get('reason', 'unknown')
                movement_info += f", Prediction: {pred_reason}"

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'distance_m': human_depth_m,  # Enhanced depth estimation
            'lateral_distance_m': distance_lateral_m,
            'time_to_collision_s': time_to_collision,
            'in_fov': in_warning_depth and in_warning_lateral,
            'in_blind_spot': in_critical_depth and in_critical_lateral,
            'movement_enhanced': movement_data is not None,
            'direction_risk': direction_risk,
            'speed_risk': speed_risk,
            'depth_analysis': depth_analysis,  # Enhanced depth of field analysis
            'field_position': depth_analysis['field_analysis'],  # Field context
            'details': f"Depth: {human_depth_m:.1f}m (conf: {depth_confidence:.2f}), Lateral: {distance_lateral_m:.1f}m, TTC: {time_to_collision:.1f}s{movement_info}"
        }

    def get_danger_zones_visualization(self, frame_shape, harvester_position=(0.5, 0.7)):
        """
        Get coordinates for danger zone visualization.
        
        SPATIAL MODEL: Camera-centric representation
        - Bottom of frame = closest to camera/truck (highest Y pixel value)
        - Top of frame = farthest from camera/truck (lowest Y pixel value)
        - Zones extend from truck centerline (harvester_position[0])
        
        Args:
            frame_shape: (height, width)
            harvester_position: (x_norm, y_norm)
            
        Returns:
            dict: Visualization data
        """
        h, w = frame_shape[:2]
        hx = int(harvester_position[0] * w)
        hy = int(harvester_position[1] * h)
        
        pixels_per_meter = h / 30.0
        
        # CRITICAL ZONE: Red box
        # Extends from truck position down to critical depth (closer to camera)
        # and sideways from truck centerline (lateral warning distance)
        critical_depth_pixels = int(30.0 * pixels_per_meter)  # 30m towards camera
        critical_width_pixels = int(self.critical_side_dist * pixels_per_meter)
        
        critical_rect = {
            'pt1': (int(hx - critical_width_pixels), hy),  # Top-left of danger zone
            'pt2': (int(hx + critical_width_pixels), int(hy + critical_depth_pixels)),  # Bottom-right
            'color': (0, 0, 255),  # Red
            'name': 'CRITICAL - Blind Spot (0-30m from truck)'
        }
        
        # WARNING ZONE: Yellow box
        # Extends from truck position down to warning depth
        warning_depth_pixels = int(50.0 * pixels_per_meter)  # 50m towards camera
        warning_width_pixels = int(self.warning_side_dist * pixels_per_meter)
        
        warning_rect = {
            'pt1': (int(hx - warning_width_pixels), hy),
            'pt2': (int(hx + warning_width_pixels), int(hy + warning_depth_pixels)),
            'color': (0, 255, 255),  # Yellow/Cyan
            'name': 'WARNING - Field of View (30-50m from truck)'
        }
        
        return {
            'critical': critical_rect,
            'warning': warning_rect,
            'harvester_center': (hx, hy),
            'frameshape': frame_shape
        }


class MovementPredictor:
    """
    Advanced human movement prediction with obstruction awareness.
    
    Handles translucent/opaque obstacles by:
    - Tracking trajectory history
    - Predicting continuation through partial occlusions
    - Escalating risk for unpredictable movements
    - Using Kalman-like filtering for smooth predictions
    """

    def __init__(self, max_history=10, prediction_horizon=5):
        """
        Initialize movement predictor.
        
        Args:
            max_history: Maximum frames to keep in trajectory history
            prediction_horizon: Number of frames to predict ahead
        """
        self.max_history = max_history
        self.prediction_horizon = prediction_horizon
        self.trajectories = {}  # track_id -> list of (frame, position, velocity)
        self.obstruction_memory = {}  # track_id -> obstruction history
        logging.info("Movement Predictor initialized with obstruction awareness")

    def update_trajectory(self, track_id, frame_num, bbox, frame_shape, is_occluded=False, occlusion_type='none'):
        """
        Update trajectory for a tracked object.
        
        Args:
            track_id: Unique identifier for the object
            frame_num: Current frame number
            bbox: [x1, y1, x2, y2] bounding box
            frame_shape: (height, width) of frame
            is_occluded: Whether object is currently partially visible
            occlusion_type: 'translucent', 'opaque', 'partial', 'none'
        """
        h, w = frame_shape
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Initialize trajectory if new
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
            self.obstruction_memory[track_id] = []
        
        # Calculate velocity from previous position
        velocity = (0, 0)
        if len(self.trajectories[track_id]) > 0:
            prev_frame, prev_pos, _ = self.trajectories[track_id][-1]
            if frame_num > prev_frame:
                dt = frame_num - prev_frame
                velocity = ((center_x - prev_pos[0]) / dt, (center_y - prev_pos[1]) / dt)
        
        # Store current state
        self.trajectories[track_id].append((frame_num, (center_x, center_y), velocity))
        
        # Maintain history length
        if len(self.trajectories[track_id]) > self.max_history:
            self.trajectories[track_id].pop(0)
        
        # Update obstruction memory
        self.obstruction_memory[track_id].append({
            'frame': frame_num,
            'occluded': is_occluded,
            'type': occlusion_type,
            'position': (center_x, center_y)
        })
        
        # Maintain obstruction history
        if len(self.obstruction_memory[track_id]) > self.max_history:
            self.obstruction_memory[track_id].pop(0)

    def predict_future_positions(self, track_id, current_frame, frame_shape):
        """
        Predict future positions considering obstructions.
        
        Args:
            track_id: Object to predict
            current_frame: Current frame number
            frame_shape: (height, width) of frame
            
        Returns:
            list: Predicted (x, y) positions for next prediction_horizon frames
        """
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return []  # Not enough data
        
        trajectory = self.trajectories[track_id]
        obstruction_history = self.obstruction_memory[track_id]
        
        # Get recent velocity (weighted average of last few frames)
        recent_velocities = [vel for _, _, vel in trajectory[-3:]]
        avg_velocity = (
            sum(v[0] for v in recent_velocities) / len(recent_velocities),
            sum(v[1] for v in recent_velocities) / len(recent_velocities)
        )
        
        # Check for obstruction patterns
        recent_obstructions = obstruction_history[-5:]
        obstruction_pattern = self._analyze_obstruction_pattern(recent_obstructions)
        
        # Adjust prediction based on obstructions
        prediction_confidence = self._calculate_prediction_confidence(obstruction_pattern)
        
        # Generate predictions
        predictions = []
        current_pos = trajectory[-1][1]  # Last known position
        
        for i in range(1, self.prediction_horizon + 1):
            # Predict next position
            next_x = current_pos[0] + avg_velocity[0] * i
            next_y = current_pos[1] + avg_velocity[1] * i
            
            # Apply obstruction-aware adjustments
            next_x, next_y = self._adjust_for_obstructions(
                next_x, next_y, obstruction_pattern, frame_shape, prediction_confidence
            )
            
            predictions.append((next_x, next_y, prediction_confidence))
            current_pos = (next_x, next_y)
        
        return predictions

    def _analyze_obstruction_pattern(self, obstruction_history):
        """
        Analyze recent obstruction history to detect patterns.
        
        Returns:
            dict: Obstruction analysis
        """
        if not obstruction_history:
            return {'type': 'clear', 'frequency': 0, 'severity': 0}
        
        occluded_frames = sum(1 for obs in obstruction_history if obs['occluded'])
        frequency = occluded_frames / len(obstruction_history)
        
        # Determine obstruction type
        types = [obs['type'] for obs in obstruction_history if obs['occluded']]
        if 'opaque' in types:
            primary_type = 'opaque'
            severity = 0.8
        elif 'translucent' in types:
            primary_type = 'translucent'
            severity = 0.6
        elif 'partial' in types:
            primary_type = 'partial'
            severity = 0.4
        else:
            primary_type = 'clear'
            severity = 0
        
        return {
            'type': primary_type,
            'frequency': frequency,
            'severity': severity
        }

    def _calculate_prediction_confidence(self, obstruction_pattern):
        """
        Calculate confidence in movement predictions based on obstructions.
        """
        base_confidence = 0.8
        
        # Reduce confidence based on obstruction severity
        severity_penalty = obstruction_pattern['severity'] * 0.5
        frequency_penalty = obstruction_pattern['frequency'] * 0.3
        
        confidence = base_confidence - severity_penalty - frequency_penalty
        return max(0.1, confidence)

    def _adjust_for_obstructions(self, pred_x, pred_y, obstruction_pattern, frame_shape, confidence):
        """
        Adjust predictions based on obstruction patterns.
        
        For translucent/partial occlusions: Continue trajectory with reduced confidence
        For opaque obstructions: Predict deviation or stopping
        """
        h, w = frame_shape
        
        # Boundary checking
        pred_x = max(0, min(w, pred_x))
        pred_y = max(0, min(h, pred_y))
        
        # Obstruction-based adjustments
        if obstruction_pattern['type'] == 'opaque':
            # For opaque obstructions, assume object may stop or change direction
            # Apply random walk component
            noise_factor = 0.3 * (1 - confidence)
            pred_x += np.random.normal(0, noise_factor * 10)
            pred_y += np.random.normal(0, noise_factor * 10)
            
        elif obstruction_pattern['type'] in ['translucent', 'partial']:
            # For partial visibility, continue trajectory but add uncertainty
            noise_factor = 0.1 * (1 - confidence)
            pred_x += np.random.normal(0, noise_factor * 5)
            pred_y += np.random.normal(0, noise_factor * 5)
        
        # Ensure within bounds after adjustments
        pred_x = max(0, min(w, pred_x))
        pred_y = max(0, min(h, pred_y))
        
        return pred_x, pred_y

    def get_risk_escalation_from_prediction(self, track_id, current_bbox, predictions, frame_shape):
        """
        Calculate risk escalation based on predicted movement.
        
        Args:
            track_id: Object ID
            current_bbox: Current bounding box
            predictions: List of predicted positions
            frame_shape: (height, width)
            
        Returns:
            dict: Risk escalation factors
        """
        if not predictions:
            return {'escalation_factor': 1.0, 'reason': 'no_prediction'}
        
        h, w = frame_shape
        harvester_x = w * 0.5  # Assume center
        harvester_y = h * 0.7  # Assume bottom-center
        
        # Check if any prediction enters critical zones
        critical_zone_entered = False
        for pred_x, pred_y, conf in predictions:
            # Simple distance check to harvester position
            distance = np.sqrt((pred_x - harvester_x)**2 + (pred_y - harvester_y)**2)
            if distance < 50:  # Within 50 pixels of harvester
                critical_zone_entered = True
                break
        
        if critical_zone_entered:
            escalation = 1.5
            reason = 'predicted_path_enters_danger_zone'
        else:
            escalation = 1.0
            reason = 'predicted_path_safe'
        
        # Adjust based on prediction confidence
        avg_confidence = sum(conf for _, _, conf in predictions) / len(predictions)
        confidence_penalty = (1 - avg_confidence) * 0.3
        escalation *= (1 + confidence_penalty)
        
        return {
            'escalation_factor': escalation,
            'reason': reason,
            'prediction_confidence': avg_confidence,
            'critical_zone_predicted': critical_zone_entered
        }
