"""
Upgraded Agricultural Safety AI System
Integrated system with all improvements:
- Upgraded detection with multi-scale inference and SAHI
- ByteTrack for robust tracking
- Dynamic safety zones with comprehensive risk assessment
- Enhanced visualization
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from collections import deque

from config import get_config, SystemConfig
from detection_upgraded import UpgradedObjectDetector
from bytetrack import ByteTrackWrapper
from safety_engine_upgraded import UpgradedSafetyEngine, RiskLevel
from visualization_upgraded import UpgradedVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpgradedAgriSafetySystem:
    """
    Complete upgraded agricultural safety AI system.
    
    Pipeline:
    Input Frame → Preprocessing → Multi-scale Detection → SAHI → 
    Tracking (ByteTrack) → Trajectory Storage → Dynamic Safety Assessment → 
    Visualization
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the upgraded safety system.
        
        Args:
            config: Optional system configuration
        """
        self.config = config if config else get_config()
        
        logger.info("Initializing Upgraded Agricultural Safety AI System")
        logger.info("=" * 60)
        
        # Initialize detection system
        logger.info("Initializing detection system...")
        self.detector = UpgradedObjectDetector(
            model_path=self.config.detection.model_path,
            config=self.config
        )
        
        # Initialize tracking system (ByteTrack)
        logger.info("Initializing ByteTrack tracker...")
        self.tracker = ByteTrackWrapper(config=self.config)
        
        # Initialize safety engine
        logger.info("Initializing safety engine...")
        self.safety_engine = UpgradedSafetyEngine(config=self.config)
        
        # Initialize visualizer
        logger.info("Initializing visualizer...")
        self.visualizer = UpgradedVisualizer(config=self.config)
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # System state
        self.system_active = True
        self.emergency_active = False
        
        logger.info("System initialization complete")
        logger.info("=" * 60)
    
    def process_frame(self, frame: np.ndarray, 
                     tractor_position: Optional[Tuple[float, float]] = None,
                     tractor_velocity: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input video frame
            tractor_position: Optional tractor position (x, y) in pixels
            tractor_velocity: Optional tractor velocity (vx, vy)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        self.frame_count += 1
        
        h, w = frame.shape[:2]
        
        # Default tractor position (center-bottom)
        if tractor_position is None:
            tractor_position = (w / 2, h * 0.85)
        
        if tractor_velocity is None:
            tractor_velocity = (0, 0)
        
        try:
            # Step 1: Detection
            detections = self.detector.detect(frame)
            
            # Step 2: Tracking with ByteTrack
            tracks = self.tracker.update(detections, frame_id=self.frame_count)
            
            # Step 3: Update safety engine with tractor state
            self.safety_engine.update_tractor_state(
                position=tractor_position,
                velocity=tractor_velocity,
                heading=-np.pi/2,  # Facing up by default
                is_reversing=False
            )
            
            # Step 4: Risk assessment (with error handling)
            try:
                risk_assessments = self.safety_engine.process_frame(tracks, frame.shape)
            except Exception as e:
                logger.error(f"Risk assessment error: {e}")
                risk_assessments = []
            
            # Step 5: Check for emergency
            self.emergency_active = any(
                ra.risk_level == RiskLevel.EMERGENCY for ra in risk_assessments
            )
            
            # Step 6: Visualization
            result_frame = frame.copy()
            
            # Draw safety zones
            zones = self.safety_engine.zone_manager.zones
            result_frame = self.visualizer.draw_safety_zones(
                result_frame, zones, tractor_position
            )
            
            # Draw tracks with risk coloring
            result_frame = self.visualizer.draw_tracks(
                result_frame, tracks, risk_assessments
            )
            
            # Draw system status
            detection_metrics = self.detector.get_evaluation_metrics()
            tracking_stats = self.tracker.get_stats()
            safety_metrics = self.safety_engine.get_metrics()
            
            # Calculate FPS
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(list(self.fps_history))
            
            result_frame = self.visualizer.draw_system_status(
                result_frame,
                detection_metrics=detection_metrics,
                tracking_stats=tracking_stats,
                safety_metrics=safety_metrics,
                fps=avg_fps
            )
            
            # Draw alert banner if emergency
            if self.emergency_active:
                result_frame = self.visualizer.draw_alert_banner(
                    result_frame, "EMERGENCY",
                    "IMMEDIATE STOP - Human in danger zone!"
                )
            
            # Compile results
            track_dicts = []
            for t in tracks:
                if hasattr(t, 'to_dict'):
                    track_dicts.append(t.to_dict())
                elif isinstance(t, dict):
                    track_dicts.append(t)
                else:
                    track_dicts.append({'track_id': -1, 'bbox': (0,0,0,0)})
            
            results = {
                'frame_number': self.frame_count,
                'processing_time': processing_time,
                'fps': avg_fps,
                'detections': detections,
                'tracks': track_dicts,
                'risk_assessments': risk_assessments,
                'emergency_active': self.emergency_active,
                'frame': result_frame,
                'metrics': {
                    'detection': detection_metrics,
                    'tracking': tracking_stats,
                    'safety': safety_metrics
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return {
                'error': str(e),
                'frame_number': self.frame_count,
                'emergency_active': True,  # Safety first
                'frame': frame
            }
    
    def run_demo(self, input_source: str = 'webcam',
                input_path: Optional[str] = None,
                max_frames: int = 0) -> None:
        """
        Run live demonstration of the system.
        
        Args:
            input_source: 'webcam' or 'video'
            input_path: Path to video file (for video input)
            max_frames: Maximum frames to process (0 = unlimited)
        """
        logger.info("Starting demo...")
        
        # Initialize video capture
        if input_source == 'webcam':
            cap = cv2.VideoCapture(0)
            logger.info("Using webcam input")
        elif input_source == 'video' and input_path:
            cap = cv2.VideoCapture(input_path)
            logger.info(f"Using video input: {input_path}")
        else:
            logger.error("Invalid input source")
            return
        
        if not cap.isOpened():
            logger.error("Could not open video source")
            return
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.system_active:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Display result
                display_frame = results.get('frame', frame)
                cv2.imshow('Agricultural Safety AI System (Upgraded)', display_frame)
                
                frame_count += 1
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Demo stopped by user")
                    break
                elif key == ord('r'):
                    self.reset()
                    logger.info("System reset")
                
                # Check frame limit
                if max_frames > 0 and frame_count >= max_frames:
                    break
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            total_time = time.time() - start_time
            logger.info("\n" + "=" * 60)
            logger.info("Demo Complete - Final Statistics:")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Total time: {total_time:.1f}s")
            logger.info(f"  Average FPS: {frame_count / total_time:.1f}")
            
            metrics = self.safety_engine.get_metrics()
            logger.info(f"  Total assessments: {metrics['total_assessments']}")
            logger.info(f"  Emergency events: {metrics['emergency_count']}")
            logger.info(f"  Critical events: {metrics['critical_count']}")
            logger.info(f"  Warning events: {metrics['warning_count']}")
            logger.info("=" * 60)
    
    def reset(self):
        """Reset system state."""
        self.detector.reset()
        self.tracker.reset()
        self.safety_engine.reset()
        self.visualizer.reset()
        self.frame_count = 0
        self.fps_history.clear()
        self.processing_times.clear()
        self.emergency_active = False
        logger.info("System reset complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'frame_count': self.frame_count,
            'fps': np.mean(list(self.fps_history)) if self.fps_history else 0,
            'emergency_active': self.emergency_active,
            'detection_metrics': self.detector.get_evaluation_metrics(),
            'tracking_stats': self.tracker.get_stats(),
            'safety_metrics': self.safety_engine.get_metrics()
        }


def main():
    """Main entry point for the upgraded system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upgraded Agricultural Safety AI System')
    parser.add_argument('--input', choices=['webcam', 'video'], default='webcam',
                       help='Input source type')
    parser.add_argument('--video-path', help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='Maximum frames to process (0 = unlimited)')
    
    args = parser.parse_args()
    
    # Initialize system
    config = get_config()
    system = UpgradedAgriSafetySystem(config)
    
    # Run demo
    system.run_demo(
        input_source=args.input,
        input_path=args.video_path,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()