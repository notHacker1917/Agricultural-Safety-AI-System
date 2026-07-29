import argparse
import cv2
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from coco_loader import COCODataset
from detection import ObjectDetector
from segmentation_tracking import DeepSORTTracker
from trajectory_storage import TrajectoryStorage
from safety_engine import SafetyEngine
from harvester_safety import HarvesterSafetyEngine, MovementPredictor
from harvester_visualizer import HarvesterSafetyVisualizer
from homography import HomographyTransformer
from stabilization import VideoStabilizer
from stabilization import OcclusionHead
from llm_risk_assessor import LLMAgriSafetyAssessor, create_scene_description, LLMProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# STEP 9: Keep it modular - encapsulate logic into functions
def compute_risk(bbox, frame_shape, obj_id, previous_ratios, risk_history, movement_data=None, trajectory=None, class_name="person"):
    """
    Compute enhanced risk score for an object with improved accuracy.
    
    Args:
        bbox: [x, y, w, h]
        frame_shape: (height, width)
        obj_id: Object ID
        previous_ratios: Dict of previous bbox ratios
        risk_history: Dict of risk score histories
        movement_data: Movement data from trajectory analysis
        trajectory: Full trajectory points
        class_name: Object class name
    
    Returns:
        dict: Risk data
    """
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]  # Handle both (h,w) and (h,w,c) formats
    bbox_height = h
    bbox_width = w
    
    # Enhanced distance estimation (inverse relationship - larger bbox = closer = higher risk)
    bbox_ratio = bbox_height / frame_height if frame_height > 0 else 0
    distance_factor = 1.0 / (bbox_ratio + 0.1)  # Avoid division by zero, normalize
    
    # Vertical position (higher = closer to ground = potentially more dangerous)
    center_y = (y + h/2) / frame_height if frame_height > 0 else 0
    vertical_risk = center_y  # Higher values = higher risk (closer to ground level)
    
    # Horizontal position (center = higher risk for harvester operations)
    center_x = (x + w/2) / frame_width if frame_width > 0 else 0.5
    lateral_offset = abs(center_x - 0.5)  # 0 = center, 0.5 = edge
    lateral_risk = 1.0 - (lateral_offset * 2)  # Center = higher risk
    
    # Enhanced temporal tracking with direction awareness
    previous_ratio = previous_ratios.get(obj_id, bbox_ratio)
    ratio_change = bbox_ratio - previous_ratio  # Positive = approaching, negative = moving away
    previous_ratios[obj_id] = bbox_ratio
    
    # Movement direction risk (approaching = much higher risk)
    if ratio_change > 0.02:  # Significantly approaching
        approach_risk = 1.0
        movement_status = "approaching_fast"
    elif ratio_change > 0.005:  # Slowly approaching
        approach_risk = 0.7
        movement_status = "approaching"
    elif ratio_change < -0.02:  # Moving away fast
        approach_risk = 0.1
        movement_status = "moving_away_fast"
    elif ratio_change < -0.005:  # Moving away slowly
        approach_risk = 0.3
        movement_status = "moving_away"
    else:  # Stationary
        approach_risk = 0.5
        movement_status = "stationary"
    
    # Velocity and acceleration from trajectory (if available)
    velocity_risk = 0.5  # Default
    acceleration_risk = 0.5  # Default
    
    if trajectory and len(trajectory) >= 3:
        # Calculate velocity (pixels per frame)
        recent_points = trajectory[-3:]
        velocities = []
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i-1][0]
            dy = recent_points[i][1] - recent_points[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            # Higher velocity = higher risk (faster movement)
            velocity_risk = min(1.0, avg_velocity / 20.0)  # Normalize
            
            # Calculate acceleration
            if len(velocities) >= 2:
                accelerations = []
                for i in range(1, len(velocities)):
                    acc = velocities[i] - velocities[i-1]
                    accelerations.append(acc)
                avg_acceleration = np.mean(accelerations)
                # Sudden acceleration = higher risk
                acceleration_risk = min(1.0, abs(avg_acceleration) / 10.0)
    
    # Situational awareness factors
    situational_risk = 0.5
    
    # Factor 1: Size consistency (sudden size changes might indicate occlusion or unusual behavior)
    if 'bbox_history' not in previous_ratios:
        previous_ratios['bbox_history'] = {}
    if obj_id not in previous_ratios['bbox_history']:
        previous_ratios['bbox_history'][obj_id] = []
    
    bbox_history = previous_ratios['bbox_history'][obj_id]
    bbox_history.append(bbox_ratio)
    if len(bbox_history) > 5:
        bbox_history.pop(0)
    
    if len(bbox_history) >= 3:
        size_variance = np.var(bbox_history)
        # High variance = potential occlusion or erratic behavior = higher risk
        size_consistency_risk = min(1.0, size_variance * 100)
        situational_risk = (situational_risk + size_consistency_risk) / 2
    
    # Factor 2: Movement pattern analysis
    if movement_data:
        direction = movement_data.get('direction', 'unknown')
        speed_category = movement_data.get('speed_category', 'moderate')
        
        # Crossing patterns are more dangerous
        if direction in ['left', 'right'] and center_y > 0.3:  # Low to ground
            crossing_risk = 0.8
        else:
            crossing_risk = 0.4
        
        # Speed categories
        speed_multipliers = {
            'very_slow': 0.3,
            'slow': 0.5,
            'moderate': 0.7,
            'fast': 0.9,
            'very_fast': 1.0
        }
        speed_risk = speed_multipliers.get(speed_category, 0.7)
        
        situational_risk = (situational_risk + crossing_risk + speed_risk) / 3
    
    # Compute comprehensive risk score with improved weighting
    base_risk = (
        0.25 * distance_factor +      # Distance (closer = higher risk)
        0.20 * vertical_risk +        # Vertical position
        0.15 * lateral_risk +         # Horizontal position (center = higher risk)
        0.15 * approach_risk +        # Movement direction
        0.10 * velocity_risk +        # Movement speed
        0.10 * acceleration_risk +    # Acceleration
        0.05 * situational_risk       # Situational factors
    )
    
    # Normalize to 0-1 range
    risk_score = min(1.0, max(0.0, base_risk))
    
    # Apply harvester context risk
    context_multiplier = get_harvester_context_risk(bbox, frame_shape, movement_status)
    risk_score *= context_multiplier
    risk_score = min(1.0, max(0.0, risk_score))  # Re-clamp after context adjustment
    
    # Enhanced temporal smoothing with trend analysis
    if obj_id not in risk_history:
        risk_history[obj_id] = []
    
    risk_history[obj_id].append(risk_score)
    if len(risk_history[obj_id]) > 5:  # Keep more history for better smoothing
        risk_history[obj_id].pop(0)
    
    # Use weighted average with recent bias (more recent = higher weight)
    if len(risk_history[obj_id]) > 1:
        weights = np.linspace(0.5, 1.0, len(risk_history[obj_id]))  # Recent bias
        weights = weights / np.sum(weights)
        smoothed_risk_score = np.average(risk_history[obj_id], weights=weights)
    else:
        smoothed_risk_score = risk_score
    
    # Trend analysis - if risk is consistently increasing, boost the score
    if len(risk_history[obj_id]) >= 3:
        recent_trend = np.polyfit(range(len(risk_history[obj_id])), risk_history[obj_id], 1)[0]
        if recent_trend > 0.01:  # Risk increasing
            trend_boost = min(0.2, recent_trend * 10)
            smoothed_risk_score = min(1.0, smoothed_risk_score + trend_boost)
    
    return {
        'risk_score': smoothed_risk_score,
        'bbox_ratio': bbox_ratio,
        'distance_factor': distance_factor,
        'vertical_risk': vertical_risk,
        'lateral_risk': lateral_risk,
        'approach_risk': approach_risk,
        'velocity_risk': velocity_risk,
        'acceleration_risk': acceleration_risk,
        'situational_risk': situational_risk,
        'movement_status': movement_status,
        'ratio_change': ratio_change
    }

def get_harvester_context_risk(bbox, frame_shape, movement_status, harvester_width_m=2.5, harvester_speed_mps=1.0):
    """
    Calculate risk based on harvester operational context.
    
    Args:
        bbox: [x, y, w, h] bounding box
        frame_shape: (height, width)
        movement_status: Current movement status
        harvester_width_m: Harvester width in meters
        harvester_speed_mps: Harvester speed in m/s
    
    Returns:
        float: Context risk multiplier (0.5-2.0)
    """
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]  # Handle both (h,w) and (h,w,c) formats
    
    # Estimate real-world position (rough approximation)
    # Assume camera is mounted on harvester looking forward
    center_x = (x + w/2) / frame_width
    center_y = (y + h/2) / frame_height
    
    # Lateral position risk (center = highest risk for harvester path)
    lateral_offset = abs(center_x - 0.5) * 2  # 0 = center, 1 = edge
    lateral_risk = 1.0 + (1.0 - lateral_offset) * 0.5  # Center gets 1.5x multiplier
    
    # Time-to-collision based on movement
    time_to_collision_risk = 1.0
    if movement_status == "approaching_fast":
        # Fast approaching = very high risk
        time_to_collision_risk = 2.0
    elif movement_status == "approaching":
        time_to_collision_risk = 1.5
    elif movement_status in ["moving_away", "moving_away_fast"]:
        # Moving away = lower risk
        time_to_collision_risk = 0.7
    elif movement_status == "stationary":
        # Stationary but in path = moderate risk
        time_to_collision_risk = 1.2
    
    # Height-based risk (taller objects might be more concerning)
    height_ratio = h / frame_height
    size_risk = 1.0 + (height_ratio - 0.1) * 2  # Taller = higher risk
    
    # Combine contextual factors
    context_risk = lateral_risk * time_to_collision_risk * size_risk
    
    # Clamp to reasonable range
    return max(0.5, min(2.0, context_risk))

def get_risk_label(score):
    """
    Map risk score to risk level with improved sensitivity.
    
    Args:
        score: Risk score (0-1)
    
    Returns:
        str: Risk level
    """
    if score > 0.75:
        return "CRITICAL"
    elif score > 0.55:
        return "HIGH_WARNING"
    elif score > 0.35:
        return "WARNING"
    elif score > 0.20:
        return "LOW_WARNING"
    else:
        return "SAFE"

def get_risk_color(risk_level):
    """Get color for 5-tier risk level."""
    """Get color for 5-tier risk level."""
    colors = {
        'SAFE': (0, 255, 0),        # Green
        'LOW': (0, 255, 255),       # Light yellow
        'MEDIUM': (0, 255, 255),    # Yellow
        'HIGH': (0, 165, 255),      # Orange
        'CRITICAL': (0, 0, 255)     # Red
    }
    return colors.get(risk_level, (255, 255, 255))  # White for unknown

def get_risk_border_thickness(risk_level):
    """Get border thickness for 5-tier risk level."""
    thicknesses = {
        'CRITICAL': 5,      # Thick red border
        'HIGH_WARNING': 4,  # Thick dark red border
        'WARNING': 3,       # Medium yellow border
        'LOW_WARNING': 2,   # Thin orange border
        'SAFE': 1           # Thin green border
    }
    return thicknesses.get(risk_level, 2)  # Default thickness

def annotate_frame(frame, tracks, trajectories, risk_results, frame_index):
    """
    Annotate frame with all overlays using 5-tier risk system.
    """
    for track in tracks:
        obj_id = track['id']
        bbox = track['bbox']
        risk = next((r for r in risk_results if r.get('id') == obj_id), None)
        if risk:
            # Use new risk level if available, else fallback to existing
            risk_level = risk.get('new_risk_level', risk.get('risk_level', 'UNKNOWN'))
            risk_score = risk.get('new_risk_score', risk.get('risk_score', 0.0))
            distance_m = risk.get('distance_m', 0.0)
            color = get_risk_color(risk_level)
            thickness = get_risk_border_thickness(risk_level)
        else:
            color = (255, 255, 255)  # White for unknown
            thickness = 2
            risk_level = "UNKNOWN"
            risk_score = 0.0
            distance_m = 0.0

        # Draw bounding box with risk-appropriate thickness
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)

        # Draw tracking ID
        cv2.putText(frame, f"ID: {obj_id}", (int(bbox[0]), int(bbox[1]) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw risk level and distance
        movement_status = risk.get('movement_status', 'unknown') if risk else 'unknown'
        llm_enhanced = risk.get('llm_enhanced', False) if risk else False
        status_short = movement_status.replace('_', ' ')
        enhancement_indicator = " [LLM]" if llm_enhanced else ""
        cv2.putText(frame, f"Risk: {risk_level} ({status_short}){enhancement_indicator}", (int(bbox[0]), int(bbox[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Distance: {distance_m:.1f}m", (int(bbox[0]), int(bbox[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Show LLM insights if available
        if llm_enhanced and risk.get('predicted_scenarios'):
            scenarios = risk.get('predicted_scenarios', [])
            if scenarios:
                cv2.putText(frame, f"LLM: {scenarios[0][:30]}...", (int(bbox[0]), int(bbox[1]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if risk and risk.get('is_occluded'):
            cv2.putText(frame, "OCCLUDED", (int(bbox[0]), int(bbox[1]) - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

        # Draw trajectory (last 30 frames)
        if obj_id in trajectories and len(trajectories[obj_id]) > 1:
            points = np.array(trajectories[obj_id][-30:], dtype=np.int32)
            cv2.polylines(frame, [points], False, color, 2)

        # Draw predicted path if available
        if risk and 'predicted_path' in risk:
            pred_path = risk['predicted_path']
            if len(pred_path) > 1:
                pred_points = np.array([(int(p[0]), int(p[1])) for p in pred_path], dtype=np.int32)
                for i in range(len(pred_points) - 1):
                    cv2.line(frame, tuple(pred_points[i]), tuple(pred_points[i+1]), color, 1, cv2.LINE_AA)

    # Add frame info
    cv2.putText(frame, f"Frame: {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add functional 3x3 grid overlay for better detection guidance
    height, width = frame.shape[:2]

    # Define tractor/equipment position (center-bottom of frame)
    tractor_x, tractor_y = width // 2, int(height * 0.85)

    # Draw 3x3 grid lines (4 vertical, 4 horizontal lines creating 9 equal squares)
    grid_color = (255, 255, 255)  # White
    grid_alpha = 0.02  # Very light overlay

    # Vertical grid lines (every 1/3 of width)
    for i in range(1, 3):  # 2 lines creating 3 columns
        x = int(width * i / 3)
        overlay = frame.copy()
        cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
        cv2.addWeighted(overlay, grid_alpha, frame, 1 - grid_alpha, 0, frame)

    # Horizontal grid lines (every 1/3 of height)
    for i in range(1, 3):  # 2 lines creating 3 rows
        y = int(height * i / 3)
        overlay = frame.copy()
        cv2.line(overlay, (0, y), (width, y), grid_color, 1)
        cv2.addWeighted(overlay, grid_alpha, frame, 1 - grid_alpha, 0, frame)

    # Draw tractor/equipment marker
    cv2.circle(frame, (tractor_x, tractor_y), 12, (255, 0, 255), 2)  # Smaller, thinner magenta circle
    cv2.putText(frame, "TRACTOR", (tractor_x - 25, tractor_y - 15),
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
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    return frame

def run_demo(input_type, input_path, max_images=100, coco_annotations=None, coco_images=None):
    """
    Run the full demo pipeline.
    """
    # Prepare outputs in a temp directory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    demo_frames_dir = os.path.join(temp_dir, 'demo_frames')
    os.makedirs(demo_frames_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, 'demo_result.mp4')

    # Prepare input based on type
    if input_type == 'video':
        # Handle video/camera input
        if input_path is None or input_path == '0':
            # Use camera
            cap = cv2.VideoCapture(0)  # Default camera
            logging.info("Using live camera input")
        else:
            # Use video file
            cap = cv2.VideoCapture(input_path)
            logging.info(f"Using video file: {input_path}")
        
        if not cap.isOpened():
            logging.error("Could not open video/camera input")
            return
        
        total_frames = max_images
        use_mock_video = False
    else:
        # Use mock video for testing 5-tier risk assessment
        use_mock_video = True
        logging.info("Using mock video frames for testing 5-tier risk assessment")
        total_frames = max_images
        cap = None

    # Initialize modules
    detector = ObjectDetector(use_mock_detections=use_mock_video)  # Use mock detections only for mock video
    tracker = DeepSORTTracker()
    storage = TrajectoryStorage()
    safety_engine = HarvesterSafetyEngine()  # Use 5-tier risk assessment
    movement_predictor = MovementPredictor()  # Movement prediction with obstruction awareness
    visualizer = HarvesterSafetyVisualizer()  # Use enhanced visualizer
    stabilizer = VideoStabilizer()
    occlusion_head = OcclusionHead()
    homography = HomographyTransformer()

    # Initialize LLM-enhanced risk assessor (optional)
    llm_assessor = None
    try:
        # Try to initialize with OpenAI first, fallback to mock
        llm_assessor = LLMAgriSafetyAssessor(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
        logging.info("LLM risk assessor initialized with OpenAI GPT-4")
    except Exception as e:
        logging.warning(f"OpenAI LLM not available: {e}")
        try:
            llm_assessor = LLMAgriSafetyAssessor(provider=LLMProvider.ANTHROPIC, model="claude-3-haiku-20240307")
            logging.info("LLM risk assessor initialized with Anthropic Claude")
        except Exception as e2:
            logging.warning(f"Anthropic LLM not available: {e2}")
            llm_assessor = LLMAgriSafetyAssessor(provider=LLMProvider.MOCK)
            logging.info("Using mock LLM risk assessor for testing")

    # Get first frame for video writer setup
    if use_mock_video:
        first_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    else:
        ret, first_frame = cap.read()
        if not ret:
            logging.error("Could not read first frame from video/camera")
            return
        first_frame = cv2.resize(first_frame, (640, 480))  # Ensure consistent size

    height, width = first_frame.shape[:2]
    video_writer = cv2.VideoWriter(video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    # Stats
    stats = defaultdict(list)
    start_time = time.time()

    print(f"Processing {total_frames} frames...")

    for frame_idx in range(total_frames):
        if use_mock_video:
            # Create test frame with humans at different distances for 5-tier risk testing
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background

            # Draw mock humans at different distances (Y-position determines distance)
            mock_humans = [
                # CRITICAL: Very close (bottom of frame) - Red zone
                ((320-30, 480-100), (320+30, 480-20), "CRITICAL", (0, 0, 255)),
                # HIGH_WARNING: Close (lower middle) - Dark red zone
                ((160-25, 240+50), (160+25, 240+120), "HIGH_WARNING", (0, 0, 200)),
                # WARNING: Medium distance (middle) - Orange zone
                ((480-20, 160), (480+20, 160+60), "WARNING", (0, 165, 255)),
                # LOW_WARNING: Far (upper middle) - Yellow zone
                ((213-15, 120), (213+15, 120+40), "LOW_WARNING", (0, 255, 255)),
                # SAFE: Very far (top) - Green zone
                ((427-10, 60), (427+10, 60+25), "SAFE", (0, 255, 0)),
            ]

            for (x1, y1), (x2, y2), risk_level, color in mock_humans:
                # Draw human bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # Draw human-like figure
                cv2.circle(frame, (int((x1+x2)//2), int(y1 + (y2-y1)//4)), int((x2-x1)//4), (255, 200, 150), -1)  # Head
                cv2.rectangle(frame, (int(x1), int(y1 + (y2-y1)//4)), (int(x2), int(y2)), (100, 150, 200), -1)  # Body
                # Label
                cv2.putText(frame, risk_level, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, "5-Tier Risk Assessment Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, "Humans at different distances", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            # Read frame from video/camera
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Could not read frame {frame_idx}, skipping")
                continue
            frame = cv2.resize(frame, (640, 480))  # Ensure consistent size
        frame_id = frame_idx

        frame_start = time.time()

        # Stabilize incoming video to reduce vibration effects before detection
        if not use_mock_video:
            frame = stabilizer.stabilize(frame)

        # Detection
        detections = detector.detect(frame)
        num_detections = len(detections)

        # Tracking
        tracks_dict = tracker.update(detections, frame)
        num_tracks = len(tracks_dict)

        # Update trajectories and compute risks
        risk_results = []
        tracks = []
        risk_distribution = {'CRITICAL': 0, 'HIGH_WARNING': 0, 'WARNING': 0, 'LOW_WARNING': 0, 'SAFE': 0}
        
        # For testing with mock detections, use original detection bboxes instead of tracked bboxes
        # since DeepSORT modifies the bounding boxes
        detection_bboxes = [bbox for bbox, conf in detections]
        
        for i, (obj_id, (bbox, mask, is_pred, is_occ, occ_dur)) in enumerate(tracks_dict.items()):
            storage.update(frame_idx, obj_id, bbox)
            trajectory = storage.get_trajectory(obj_id)

            # Use original detection bbox for risk calculation (not tracked bbox)
            risk_bbox = detection_bboxes[i] if i < len(detection_bboxes) else bbox

            # Extract movement data from trajectory for enhanced risk assessment
            movement_data = None
            if len(trajectory) >= 2:
                # Calculate movement direction and speed
                prev_pos = trajectory[-2] if len(trajectory) >= 2 else trajectory[-1]
                curr_pos = trajectory[-1]
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]

                # Determine direction (simplified)
                if abs(dy) > abs(dx):
                    direction = 'down' if dy > 0 else 'up'
                else:
                    direction = 'right' if dx > 0 else 'left'

                # Estimate speed category
                speed = np.sqrt(dx**2 + dy**2)
                if speed > 15:
                    speed_category = 'very_fast'
                elif speed > 10:
                    speed_category = 'fast'
                elif speed > 5:
                    speed_category = 'moderate'
                elif speed > 2:
                    speed_category = 'slow'
                else:
                    speed_category = 'very_slow'

                movement_data = {
                    'direction': direction,
                    'speed_category': speed_category,
                    'trajectory_pattern': {'consistency': 0.8, 'is_human_like': True}
                }

            if movement_data is None:
                movement_data = {}

            # Attach occlusion reasoning and ground-plane projection to movement context
            occlusion_info = occlusion_head.predict_occlusion(bbox, detections, is_predicted=is_pred, track_history=trajectory)
            movement_data.update({
                'is_occluded': is_occ,
                'occlusion_duration': occ_dur,
                'occlusion_confidence': occlusion_info['occlusion_confidence'],
                'occlusion_reason': occlusion_info['reason'],
            })

            if not use_mock_video:
                try:
                    ground_point = homography.project_bbox_to_ground_plane(risk_bbox)
                    movement_data['ground_point'] = ground_point
                except Exception as e:
                    logging.debug(f"Homography projection failed for bbox {risk_bbox}: {e}")

            # Update movement predictor with current track state
            occlusion_type = 'opaque' if is_occ and occ_dur > 5 else 'partial' if is_occ else 'none'
            movement_predictor.update_trajectory(obj_id, frame_idx, bbox, frame.shape[:2], 
                                               is_occluded=is_occ, occlusion_type=occlusion_type)

            # Get movement predictions and risk escalation
            predictions = movement_predictor.predict_future_positions(obj_id, frame_idx, frame.shape[:2])
            prediction_escalation = movement_predictor.get_risk_escalation_from_prediction(
                obj_id, risk_bbox, predictions, frame.shape[:2])
            
            # Include prediction data in movement context
            movement_data.update({
                'predictions': predictions,
                'prediction_escalation': prediction_escalation
            })

            risk = safety_engine.compute_risk_level(risk_bbox, frame.shape, movement_data=movement_data)
            risk['id'] = obj_id
            risk['is_occluded'] = is_occ
            
            # STEP 2-10: Add new risk estimation module
            # Compute new features per object
            if not hasattr(run_demo, 'previous_ratios'):
                run_demo.previous_ratios = {}
            if not hasattr(run_demo, 'risk_history'):
                run_demo.risk_history = {}
            
            new_risk_data = compute_risk(risk_bbox, frame.shape, obj_id, run_demo.previous_ratios, run_demo.risk_history, movement_data, trajectory)
            new_risk_level = get_risk_label(new_risk_data['risk_score'])
            
            # LLM-enhanced risk assessment (optional)
            enhanced_risk_data = None
            if llm_assessor and frame_idx % 5 == 0:  # Run LLM analysis every 5 frames to avoid API limits
                try:
                    # Prepare scene description for LLM
                    human_data = [{
                        'position': f"bbox_{risk_bbox}",
                        'distance_m': new_risk_data.get('distance_factor', 10),
                        'movement': new_risk_data.get('movement_status', 'stationary'),
                        'current_risk': new_risk_level,
                        'bbox_ratio': new_risk_data.get('bbox_ratio', 0)
                    }]
                    
                    movement_patterns = [new_risk_data.get('movement_status', 'stationary')]
                    scene_desc = create_scene_description(
                        num_humans=1,  # Per object analysis
                        human_data=human_data,
                        movement_patterns=movement_patterns,
                        environmental_factors=["field_operation", "daylight", "good_visibility"]
                    )
                    
                    # Traditional risk data for LLM enhancement
                    traditional_data = {
                        'risk_level': new_risk_level,
                        'risk_score': new_risk_data['risk_score'],
                        'reasoning': f"CV analysis: {new_risk_data.get('movement_status', 'stationary')} human at distance factor {new_risk_data.get('distance_factor', 1):.2f}"
                    }
                    
                    # Get enhanced assessment
                    enhanced_risk_data = llm_assessor.enhance_risk_assessment(traditional_data, scene_desc)
                    
                    # Use enhanced results if available
                    if enhanced_risk_data:
                        new_risk_level = enhanced_risk_data['risk_level']
                        new_risk_data['risk_score'] = enhanced_risk_data['risk_score']
                        new_risk_data['llm_reasoning'] = enhanced_risk_data.get('reasoning', '')
                        new_risk_data['predicted_scenarios'] = enhanced_risk_data.get('predicted_scenarios', [])
                        new_risk_data['recommended_actions'] = enhanced_risk_data.get('recommended_actions', [])
                        
                        logging.info(f"LLM Enhanced Risk: {new_risk_level} (score: {new_risk_data['risk_score']:.3f})")
                        
                except Exception as e:
                    logging.warning(f"LLM enhancement failed: {e}. Using traditional assessment.")
            
            # Update risk dict with enhanced module data
            risk['new_risk_level'] = new_risk_level
            risk['new_risk_score'] = new_risk_data['risk_score']
            risk['movement_status'] = new_risk_data.get('movement_status', 'unknown')
            risk['distance_factor'] = new_risk_data.get('distance_factor', 0)
            
            # Add LLM data if available
            if enhanced_risk_data:
                risk['llm_enhanced'] = True
                risk['llm_reasoning'] = enhanced_risk_data.get('reasoning', '')
                risk['predicted_scenarios'] = enhanced_risk_data.get('predicted_scenarios', [])
                risk['recommended_actions'] = enhanced_risk_data.get('recommended_actions', [])
            else:
                risk['llm_enhanced'] = False
            
            risk_results.append(risk)
            tracks.append({'id': obj_id, 'bbox': bbox})
            risk_distribution[risk.get('risk_level', 'UNKNOWN')] += 1

        annotated_frame = annotate_frame(frame, tracks, {t['id']: storage.get_trajectory(t['id']) for t in tracks}, risk_results, frame_idx)
        frame_path = os.path.join(demo_frames_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(frame_path, annotated_frame)

        # Save frame data for interface analysis
        frame_data = {
            'frame_idx': int(frame_idx),
            'timestamp': float(time.time()),
            'num_detections': int(len(risk_results)),
            'risks': risk_results,
            'llm_enhanced': bool(any(r.get('llm_enhanced', False) for r in risk_results))
        }
        
        # Add LLM data if any risk has it
        for risk in risk_results:
            if risk.get('llm_enhanced', False):
                frame_data.update({
                    'llm_reasoning': str(risk.get('llm_reasoning', '')),
                    'predicted_scenarios': list(risk.get('predicted_scenarios', [])),
                    'recommended_actions': list(risk.get('recommended_actions', [])),
                    'new_risk_level': str(risk.get('new_risk_level', 'SAFE')),
                    'llm_confidence': float(risk.get('llm_confidence', 0))
                })
                break  # Use first LLM-enhanced risk data
        
        frame_data_path = os.path.join(demo_frames_dir, f"frame_{frame_idx:06d}_data.json")
        try:
            with open(frame_data_path, 'w') as f:
                json.dump(frame_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save frame data: {e}")

        # Write to video
        video_writer.write(annotated_frame)

        # Update stats
        frame_time = time.time() - frame_start
        fps = 1 / frame_time if frame_time > 0 else 0
        stats['fps'].append(fps)
        stats['detections'].append(num_detections)
        stats['tracks'].append(num_tracks)
        stats['risk_dist'].append(risk_distribution)

    video_writer.release()

    # Cleanup video capture if used
    if cap is not None:
        cap.release()

    # Final stats
    total_time = time.time() - start_time
    avg_fps = np.mean(stats['fps']) if len(stats['fps']) > 0 else 0.0
    total_detections = sum(stats['detections'])
    total_tracks = sum(stats['tracks'])
    avg_risk_dist = {
        'SAFE': np.mean([d.get('SAFE', 0) for d in stats['risk_dist']]) if len(stats['risk_dist']) > 0 else 0.0,
        'LOW_WARNING': np.mean([d.get('LOW_WARNING', 0) for d in stats['risk_dist']]) if len(stats['risk_dist']) > 0 else 0.0,
        'WARNING': np.mean([d.get('WARNING', 0) for d in stats['risk_dist']]) if len(stats['risk_dist']) > 0 else 0.0,
        'HIGH_WARNING': np.mean([d.get('HIGH_WARNING', 0) for d in stats['risk_dist']]) if len(stats['risk_dist']) > 0 else 0.0,
        'CRITICAL': np.mean([d.get('CRITICAL', 0) for d in stats['risk_dist']]) if len(stats['risk_dist']) > 0 else 0.0,
    }

    print("\nDemo completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracked objects: {total_tracks}")
    stats_output_path = os.path.join(temp_dir, 'demo_stats.json')
    stats_payload = {
        'frame_count': len(stats['fps']),
        'total_time_s': total_time,
        'average_fps': avg_fps,
        'total_detections': total_detections,
        'total_tracks': total_tracks,
        'average_risk_distribution': avg_risk_dist,
        'per_frame_detections': stats['detections'],
        'per_frame_tracks': stats['tracks'],
        'per_frame_risk_distribution': stats['risk_dist'],
    }
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_payload, f, indent=2)

    print("\nDemo completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracked objects: {total_tracks}")
    print(f"Average risk distribution: {avg_risk_dist}")
    print(f"Outputs saved to: {temp_dir}")
    print(f"Stats saved to: {stats_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Agricultural Safety AI Demo Runner')
    parser.add_argument('--input-type', choices=['coco', 'video'], required=True,
                        help='Input type: coco or video')
    parser.add_argument('--input-path', help='Path to video file (for video input). If not provided, uses sample_video.mp4 if present.')
    parser.add_argument('--coco-annotations', help='Path to COCO annotations JSON (for coco input)')
    parser.add_argument('--coco-images', help='Path to COCO images directory (for coco input)')
    parser.add_argument('--max-images', type=int, default=100,
                        help='Maximum number of images/frames to process (default: 100)')

    args = parser.parse_args()

    run_demo(args.input_type, args.input_path, args.max_images,
             args.coco_annotations, args.coco_images)

if __name__ == '__main__':
    main()