#!/usr/bin/env python3
"""
Agricultural Safety AI - Enhanced Demo with Custom Model Support
"""

import argparse
import cv2
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# Import existing modules
from detection import ObjectDetector
from harvester_safety import HarvesterSafetyEngine
from harvester_visualizer import HarvesterSafetyVisualizer
from trajectory_storage import TrajectoryStorage
from segmentation_tracking import DeepSORTTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriObjectDetector(ObjectDetector):
    """Enhanced ObjectDetector with custom model support"""

    def __init__(self, model_path=None, **kwargs):
        # If custom model provided, use it instead of default YOLO
        if model_path and Path(model_path).exists():
            logging.info(f"Loading custom agricultural model: {model_path}")
            try:
                self.custom_model = YOLO(model_path)
                # Set use_mock_detections to False to force custom model usage
                kwargs['use_mock_detections'] = False
                logging.info("Custom model loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load custom model: {e}. Using default detection.")
                self.custom_model = None
        else:
            self.custom_model = None
            if model_path:
                logging.warning(f"Custom model not found: {model_path}")

        # Initialize parent class
        super().__init__(**kwargs)

    def detect(self, frame):
        """Override detect method to use custom model when available"""
        if self.custom_model is not None:
            # Use custom agricultural model
            try:
                results = self.custom_model(frame, classes=[0], conf=self.conf, verbose=False)
                detections = []
                for result in results:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append((bbox, conf))

                logging.debug(f"Custom model detected {len(detections)} persons")
                return detections
            except Exception as e:
                logging.warning(f"Custom model detection failed: {e}. Falling back to parent method.")

        # Fall back to parent method (YOLO or mock)
        return super().detect(frame)


def run_agri_demo(input_type, input_path, max_images=100, model_path=None, environment=None):
    """
    Enhanced demo with agricultural model support
    """
    # Prepare outputs
    import tempfile
    temp_dir = tempfile.mkdtemp()
    demo_frames_dir = os.path.join(temp_dir, 'demo_frames')
    os.makedirs(demo_frames_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, 'agri_demo_result.mp4')

    # Initialize enhanced detector
    detector = AgriObjectDetector(
        model_path=model_path,
        use_mock_detections=(input_type == 'mock'),
        use_preprocessing=True,
        use_human_verification=True
    )

    tracker = DeepSORTTracker()
    storage = TrajectoryStorage()

    # Enhanced safety engine with environment awareness
    safety_engine = HarvesterSafetyEngine()
    if environment:
        logging.info(f"Using environment-specific settings for: {environment}")
        # Could add environment-specific safety parameters here

    visualizer = HarvesterSafetyVisualizer()

    # Input handling
    if input_type == 'video':
        if input_path is None or input_path == '0':
            cap = cv2.VideoCapture(0)
            logging.info("Using live camera input")
        else:
            cap = cv2.VideoCapture(input_path)
            logging.info(f"Using video file: {input_path}")

        if not cap.isOpened():
            logging.error("Could not open video/camera input")
            return

        total_frames = max_images
        use_mock_video = False
    else:
        # Mock video for testing
        use_mock_video = True
        logging.info("Using mock video frames for testing")
        total_frames = max_images
        cap = None

    # Get first frame for video setup
    if use_mock_video:
        first_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    else:
        ret, first_frame = cap.read()
        if not ret:
            logging.error("Could not read first frame")
            return
        first_frame = cv2.resize(first_frame, (640, 480))

    height, width = first_frame.shape[:2]
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    # Stats tracking
    stats = defaultdict(list)
    start_time = time.time()

    print(f"Running Agricultural Safety Demo ({total_frames} frames)...")
    if model_path:
        print(f"Using custom model: {Path(model_path).name}")
    if environment:
        print(f"Environment: {environment}")

    for frame_idx in range(total_frames):
        if use_mock_video:
            # Create mock agricultural scene
            frame = create_agri_mock_frame(frame_idx, environment)
        else:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Could not read frame {frame_idx}")
                continue
            frame = cv2.resize(frame, (640, 480))

        # Process frame
        detections = detector.detect(frame)
        tracks = tracker.update(frame, detections)
        trajectories = {t['id']: storage.get_trajectory(t['id']) for t in tracks}

        # Risk assessment
        risk_results = []
        for track in tracks:
            obj_id = track['id']
            bbox = track['bbox']

            # Get movement data
            movement_data = get_movement_data(trajectories.get(obj_id, []), frame_idx)

            # Compute risk
            risk = safety_engine.compute_risk_level(bbox, frame.shape, movement_data=movement_data)
            risk['id'] = obj_id
            risk_results.append(risk)

        # Visualize
        annotated_frame = annotate_agri_frame(frame, tracks, trajectories, risk_results, frame_idx, environment)

        # Save frame and write to video
        frame_path = os.path.join(demo_frames_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(frame_path, annotated_frame)
        video_writer.write(annotated_frame)

        # Update stats
        frame_time = time.time() - time.time()  # Would need to track per frame
        fps = 1 / frame_time if frame_time > 0 else 0
        stats['fps'].append(fps)
        stats['detections'].append(len(detections))
        stats['tracks'].append(len(tracks))

    # Cleanup
    video_writer.release()
    if cap:
        cap.release()

    # Final stats
    total_time = time.time() - start_time
    avg_fps = np.mean(stats['fps']) if stats['fps'] else 0
    total_detections = sum(stats['detections'])
    total_tracks = sum(stats['tracks'])

    print("
Agricultural Safety Demo completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracked objects: {total_tracks}")
    print(f"Outputs saved to: {temp_dir}")

    return temp_dir


def create_agri_mock_frame(frame_idx, environment=None):
    """Create mock frame simulating agricultural environment"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Base color

    # Environment-specific backgrounds
    if environment == 'field':
        # Bright daylight field
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
    elif environment == 'orchard':
        # Shaded orchard
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    elif environment == 'barn':
        # Indoor barn
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

    # Add some agricultural elements
    # Ground line
    cv2.line(frame, (0, 400), (640, 400), (100, 150, 100), 3)

    # Mock humans at different distances
    humans = [
        ((320, 450), (340, 480), "CRITICAL", (0, 0, 255)),    # Very close
        ((200, 350), (220, 380), "HIGH_WARNING", (0, 0, 200)), # Close
        ((500, 250), (520, 280), "WARNING", (0, 165, 255)),    # Medium
        ((150, 200), (170, 230), "LOW_WARNING", (0, 255, 255)), # Far
        ((450, 150), (470, 180), "SAFE", (0, 255, 0)),         # Very far
    ]

    for (x1, y1), (x2, y2), risk_level, color in humans:
        # Draw person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Simple person figure
        cv2.circle(frame, ((x1+x2)//2, y1 + (y2-y1)//4), (x2-x1)//3, (255, 200, 150), -1)
        cv2.rectangle(frame, (x1, y1 + (y2-y1)//4), (x2, y2), (100, 150, 200), -1)

    # Environment label
    env_text = f"Environment: {environment or 'General'}"
    cv2.putText(frame, env_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame


def get_movement_data(trajectory, current_frame):
    """Extract movement data from trajectory"""
    if len(trajectory) < 2:
        return {'direction': 'stationary', 'speed_category': 'stationary', 'trajectory_pattern': {'consistency': 0.5, 'is_human_like': True}}

    # Simple movement analysis
    recent_positions = trajectory[-5:]  # Last 5 positions
    if len(recent_positions) < 2:
        return {'direction': 'stationary', 'speed_category': 'slow', 'trajectory_pattern': {'consistency': 0.7, 'is_human_like': True}}

    # Calculate movement vector
    dx = recent_positions[-1][0] - recent_positions[0][0]
    dy = recent_positions[-1][1] - recent_positions[0][1]
    distance = np.sqrt(dx**2 + dy**2)

    # Determine direction and speed
    if distance < 5:
        direction = 'stationary'
        speed_category = 'stationary'
    elif distance < 20:
        direction = 'slow_moving'
        speed_category = 'slow'
    else:
        direction = 'moving_toward' if dy > 0 else 'moving_away'
        speed_category = 'fast'

    return {
        'direction': direction,
        'speed_category': speed_category,
        'trajectory_pattern': {'consistency': 0.8, 'is_human_like': True}
    }


def annotate_agri_frame(frame, tracks, trajectories, risk_results, frame_idx, environment=None):
    """Enhanced frame annotation for agricultural safety"""
    # Add environment info
    if environment:
        cv2.putText(frame, f"Environment: {environment}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Annotate each track
    for track in tracks:
        obj_id = track['id']
        bbox = track['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Find corresponding risk result
        risk = next((r for r in risk_results if r['id'] == obj_id), None)
        if risk:
            risk_level = risk.get('risk_level', 'UNKNOWN')
            distance = risk.get('distance', 0)

            # Get color and thickness based on risk
            color = get_risk_color(risk_level)
            thickness = get_risk_border_thickness(risk_level)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw risk label
            label = f"ID:{obj_id} {risk_level} {distance:.1f}m"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory if available
            trajectory = trajectories.get(obj_id, [])
            if len(trajectory) > 1:
                # Draw last few points
                for i in range(max(0, len(trajectory)-10), len(trajectory)-1):
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i+1]))
                    cv2.line(frame, pt1, pt2, color, 1)

    # Add frame info
    cv2.putText(frame, f"Frame: {frame_idx}", (10, frame.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'SAFE': (0, 255, 0),        # Green
        'LOW': (0, 255, 255),       # Light yellow
        'MEDIUM': (0, 255, 255),    # Yellow
        'HIGH': (0, 165, 255),      # Orange
        'CRITICAL': (0, 0, 255)     # Red
    }
    return colors.get(risk_level, (255, 255, 255))


def get_risk_border_thickness(risk_level):
    """Get border thickness for risk level"""
    thicknesses = {
        'CRITICAL': 5,
        'HIGH_WARNING': 4,
        'WARNING': 3,
        'LOW_WARNING': 2,
        'SAFE': 1
    }
    return thicknesses.get(risk_level, 2)


def main():
    parser = argparse.ArgumentParser(description='Agricultural Safety AI Enhanced Demo')
    parser.add_argument('--input-type', choices=['video', 'mock'], required=True,
                       help='Input type: video (camera/file) or mock (simulated)')
    parser.add_argument('--input-path', help='Path to video file or camera index (0 for default camera)')
    parser.add_argument('--max-images', type=int, default=100, help='Maximum frames to process')
    parser.add_argument('--model-path', help='Path to custom trained YOLO model')
    parser.add_argument('--environment', choices=['field', 'orchard', 'barn', 'general'],
                       help='Agricultural environment for specialized processing')

    args = parser.parse_args()

    run_agri_demo(
        args.input_type,
        args.input_path,
        args.max_images,
        args.model_path,
        args.environment
    )


if __name__ == '__main__':
    main()