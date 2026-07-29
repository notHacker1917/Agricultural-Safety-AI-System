#!/usr/bin/env python3
"""
Agricultural Harvester Safety System Demo.

Demonstrates:
1. Multi-method human detection in challenging field conditions
2. Harvester field-of-view and blind-spot analysis
3. Real-time safety alerts for operators
"""

import cv2
import logging
import argparse
import tempfile
import json
from datetime import datetime
import numpy as np

from agri_detector import AgriculturalHumanDetector
from harvester_safety import HarvesterSafetyEngine
from harvester_visualizer import HarvesterSafetyVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_harvester_safety_demo(input_type='video', input_path=0, max_images=20, output_dir=None):
    """
    Run harvester safety monitoring demo.
    
    Args:
        input_type: 'video', 'image', or 'webcam'
        input_path: Path to video/image or camera index
        max_images: Max frames to process
        output_dir: Output directory for results
    """
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        logging.info(f"Output directory: {output_dir}")
    
    # Initialize components
    logging.info("Initializing Agricultural Harvester Safety System...")
    
    detector = AgriculturalHumanDetector(model_path='yolov8l.pt', conf=0.45)
    safety_engine = HarvesterSafetyEngine(
        critical_forward_distance=30,
        critical_side_distance=5,
        warning_forward_distance=50,
        warning_side_distance=15
    )
    visualizer = HarvesterSafetyVisualizer()
    
    # Initialize video capture
    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
    elif input_type == 'webcam':
        cap = cv2.VideoCapture(int(input_path))
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    output_video = f"{output_dir}/harvester_safety_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    frame_dir = f"{output_dir}/safety_frames"
    import os
    os.makedirs(frame_dir, exist_ok=True)
    
    # Process frames
    frame_count = 0
    critical_events = []
    warning_events = []
    prev_frame = None
    total_detections = 0
    total_critical = 0
    total_warning = 0
    
    logging.info(f"Processing frames... (max {max_images})")
    
    while frame_count < max_images:
        ret, frame = cap.read()
        
        if not ret:
            logging.warning("Failed to read frame or end of video")
            break
        
        frame_count += 1
        
        # Detect humans using multi-method approach
        detections = detector.detect(frame, prev_frame)
        total_detections += len(detections)
        
        # Assess risk for each detection
        risk_assessments = []
        for detection in detections:
            # Handle enhanced detection format: (bbox, confidence, movement_data)
            if len(detection) == 3:
                bbox, conf, movement_data = detection
            else:
                # Legacy format: (bbox, confidence)
                bbox, conf = detection
                movement_data = None
            
            risk = safety_engine.compute_risk_level(bbox, frame.shape, movement_data=movement_data)
            risk_assessments.append(risk)
            
            if risk['risk_level'] == 'CRITICAL':
                total_critical += 1
                critical_events.append({
                    'frame': int(frame_count),
                    'position': [float(x) for x in bbox] if hasattr(bbox, '__iter__') else [float(bbox)],
                    'risk_score': float(risk['risk_score']),
                    'time_to_collision': float(risk['time_to_collision_s']),
                    'movement_enhanced': bool(risk.get('movement_enhanced', False)),
                    'direction': str(movement_data.get('direction', 'unknown') if movement_data else 'unknown'),
                    'speed': str(movement_data.get('speed_category', 'unknown') if movement_data else 'unknown'),
                    'depth_confidence': float(risk.get('depth_analysis', {}).get('depth_confidence', 0)),
                    'timestamp': datetime.now().isoformat()
                })
            elif risk['risk_level'] == 'HIGH_WARNING':
                total_critical += 1  # Count high warnings with critical for summary
                critical_events.append({
                    'frame': int(frame_count),
                    'position': [float(x) for x in bbox] if hasattr(bbox, '__iter__') else [float(bbox)],
                    'risk_score': float(risk['risk_score']),
                    'time_to_collision': float(risk['time_to_collision_s']),
                    'movement_enhanced': bool(risk.get('movement_enhanced', False)),
                    'direction': str(movement_data.get('direction', 'unknown') if movement_data else 'unknown'),
                    'speed': str(movement_data.get('speed_category', 'unknown') if movement_data else 'unknown'),
                    'depth_confidence': float(risk.get('depth_analysis', {}).get('depth_confidence', 0)),
                    'risk_level': 'HIGH_WARNING',
                    'timestamp': datetime.now().isoformat()
                })
            elif risk['risk_level'] == 'WARNING':
                total_warning += 1
                warning_events.append({
                    'frame': int(frame_count),
                    'position': [float(x) for x in bbox] if hasattr(bbox, '__iter__') else [float(bbox)],
                    'risk_score': float(risk['risk_score']),
                    'movement_enhanced': bool(risk.get('movement_enhanced', False)),
                    'direction': str(movement_data.get('direction', 'unknown') if movement_data else 'unknown'),
                    'depth_confidence': float(risk.get('depth_analysis', {}).get('depth_confidence', 0))
                })
            elif risk['risk_level'] == 'LOW_WARNING':
                total_warning += 1  # Count low warnings with warnings for summary
                warning_events.append({
                    'frame': int(frame_count),
                    'position': [float(x) for x in bbox] if hasattr(bbox, '__iter__') else [float(bbox)],
                    'risk_score': float(risk['risk_score']),
                    'movement_enhanced': bool(risk.get('movement_enhanced', False)),
                    'direction': str(movement_data.get('direction', 'unknown') if movement_data else 'unknown'),
                    'depth_confidence': float(risk.get('depth_analysis', {}).get('depth_confidence', 0)),
                    'risk_level': 'LOW_WARNING'
                })
        
        # Get visualization data
        zones_data = safety_engine.get_danger_zones_visualization(frame.shape)
        
        # Annotate frame
        annotated = visualizer.annotate_frame(frame, detections, risk_assessments, zones_data)
        
        # Add frame information
        cv2.putText(annotated, f"Frame: {frame_count}", (20, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save output
        out.write(annotated)
        
        # Save frame image
        frame_filename = f"{frame_dir}/frame_{frame_count:06d}.jpg"
        cv2.imwrite(frame_filename, annotated)
        
        prev_frame = frame.copy()
        
        if frame_count % 10 == 0:
            logging.info(f"Processed {frame_count} frames | Detections: {total_detections} | Critical: {total_critical}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save report
    report = {
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'critical_events': total_critical,
        'warning_events': total_warning,
        'critical_event_details': critical_events[:10],  # First 10 for brevity
        'timestamp': datetime.now().isoformat(),
        'system': 'Agricultural Harvester Safety System'
    }
    
    report_file = f"{output_dir}/safety_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"Harvester Safety Demo Completed!")
    logging.info(f"{'='*60}")
    logging.info(f"Frames processed: {frame_count}")
    logging.info(f"Total human detections: {total_detections}")
    logging.info(f"Critical zone incidents: {total_critical}")
    logging.info(f"Warning zone incidents: {total_warning}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Video saved: {output_video}")
    logging.info(f"Report saved: {report_file}")
    logging.info(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Agricultural Harvester Safety System Demo")
    parser.add_argument('--input-type', default='video', choices=['video', 'webcam'],
                       help='Input source type')
    parser.add_argument('--input-path', default='0', help='Path to video or camera index')
    parser.add_argument('--max-images', type=int, default=20, help='Maximum frames to process')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    
    args = parser.parse_args()
    
    run_harvester_safety_demo(
        input_type=args.input_type,
        input_path=args.input_path,
        max_images=args.max_images,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
