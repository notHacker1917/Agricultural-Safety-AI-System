"""
OPTIMIZED DETECTION INTEGRATION FOR DEMO
Runs advanced detection methods on demo frames with visual comparison
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import json

from detection import ObjectDetector  # Base YOLO detector
from advanced_detection_algorithms import (
    EnsembleHumanDetector,
    MultiScaleHumanDetector,
    MotionBasedDetector,
    DepthBasedDetector,
    RobustPreprocessor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedDemoProcessor:
    """Process demo frames with optimized detection algorithms"""
    
    def __init__(self, use_webcam: bool = True, video_path: str = None):
        """
        Initialize demo processor
        
        Args:
            use_webcam: Use webcam if True, else use video file
            video_path: Path to video file if not using webcam
        """
        logger.info("=" * 80)
        logger.info("OPTIMIZED DETECTION DEMO PROCESSOR")
        logger.info("=" * 80)
        
        # Initialize base detector
        logger.info("\n1. Initializing Base YOLO Detector...")
        try:
            self.base_detector = ObjectDetector(
                model_path='yolov8n.pt',
                conf=0.5,
                use_preprocessing=True,
                use_human_verification=True
            )
            logger.info("   ✓ Base detector ready")
        except Exception as e:
            logger.error(f"   ✗ Failed to initialize detector: {e}")
            raise
        
        # Initialize ensemble detector
        logger.info("\n2. Initializing Ensemble Detector...")
        try:
            self.ensemble_detector = EnsembleHumanDetector(
                base_yolo_detector=self.base_detector,
                use_motion=True,
                use_depth=True,
                use_context=True
            )
            logger.info("   ✓ Ensemble detector ready")
        except Exception as e:
            logger.error(f"   ✗ Failed to initialize ensemble: {e}")
            raise
        
        # Initialize individual detectors for comparison
        logger.info("\n3. Initializing Individual Detectors...")
        self.motion_detector = MotionBasedDetector()
        self.depth_analyzer = DepthBasedDetector()
        self.preprocessor = RobustPreprocessor()
        logger.info("   ✓ All detectors initialized")
        
        # Video source
        self.use_webcam = use_webcam
        self.video_path = video_path
        self.cap = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'detections_by_method': {},
            'processing_times': [],
            'critical_detections': 0,
            'danger_detections': 0
        }
    
    def run_demo(self, max_frames: int = 50, show_visualization: bool = True):
        """
        Run optimized detection on demo frames
        
        Args:
            max_frames: Maximum number of frames to process
            show_visualization: Show real-time visualization
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING DEMO PROCESSING")
        logger.info("=" * 80)
        
        # Initialize video source
        if self.use_webcam:
            logger.info("\n➤ Using webcam (device 0)")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            logger.info(f"\n➤ Using video file: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error("✗ Failed to open video source")
            return
        
        frame_count = 0
        
        try:
            while frame_count < max_frames:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("\n✓ End of video stream")
                    break
                
                frame_count += 1
                
                logger.info(f"\n┌─ FRAME {frame_count}/{max_frames}")
                logger.info(f"│  Size: {frame.shape[1]}x{frame.shape[0]}")
                
                # Process frame with advanced detection
                start_time = time.time()
                
                try:
                    detection_results = self._process_frame_advanced(frame)
                    processing_time = time.time() - start_time
                    self.stats['processing_times'].append(processing_time)
                    
                    logger.info(f"│  Processing time: {processing_time*1000:.1f}ms")
                    logger.info(f"│  ├─ Base YOLO: {detection_results['base_count']} detections")
                    logger.info(f"│  ├─ Motion: {detection_results['motion_count']} detections")
                    logger.info(f"│  ├─ Ensemble: {detection_results['ensemble_count']} detections")
                    
                    # Show visualization
                    if show_visualization:
                        visualization = self._create_visualization(
                            frame, detection_results
                        )
                        
                        # Display
                        cv2.imshow('Optimized Detection Demo', visualization)
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("\n✓ User quit")
                            break
                    
                    # Update statistics
                    self.stats['frames_processed'] += 1
                    self.stats['total_detections'] += detection_results['ensemble_count']
                    
                    # Track risk levels
                    for det in detection_results.get('ensemble_detections', []):
                        if det.get('category') == 'CRITICAL':
                            self.stats['critical_detections'] += 1
                        elif det.get('category') == 'DANGER':
                            self.stats['danger_detections'] += 1
                    
                    logger.info(f"└─ (Running avg: {np.mean(self.stats['processing_times'])*1000:.1f}ms/frame)")
                
                except Exception as e:
                    logger.error(f"│  ✗ Error processing frame: {e}")
                    logger.info("└─")
                    continue
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_statistics()
    
    def _process_frame_advanced(self, frame: np.ndarray) -> Dict:
        """Process single frame with multiple detection methods"""
        
        results = {
            'frame': frame.copy(),
            'base_detections': [],
            'base_count': 0,
            'motion_detections': [],
            'motion_count': 0,
            'ensemble_detections': [],
            'ensemble_count': 0
        }
        
        # 1. Base YOLO detection
        try:
            base_dets = self.base_detector.detect(frame)
            results['base_detections'] = base_dets if isinstance(base_dets, list) else []
            results['base_count'] = len(results['base_detections'])
        except Exception as e:
            logger.debug(f"   Base detection error: {e}")
        
        # 2. Motion detection
        try:
            motion_dets = self.motion_detector.detect_motion(frame)
            results['motion_detections'] = motion_dets
            results['motion_count'] = len(motion_dets)
        except Exception as e:
            logger.debug(f"   Motion detection error: {e}")
        
        # 3. Ensemble detection
        try:
            ensemble_dets = self.ensemble_detector.detect(frame)
            results['ensemble_detections'] = ensemble_dets
            results['ensemble_count'] = len(ensemble_dets)
        except Exception as e:
            logger.debug(f"   Ensemble detection error: {e}")
        
        return results
    
    def _create_visualization(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Create visualization comparing detection methods"""
        
        h, w = frame.shape[:2]
        
        # Create split view: Original | Ensemble
        vis = np.hstack([frame.copy(), frame.copy()])
        
        # Draw base detections on left side
        for det in results['base_detections']:
            if isinstance(det, (tuple, list)) and len(det) >= 2:
                bbox, conf = det[0], det[1]
                if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f'YOLO {conf:.2f}', (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ensemble detections on right side
        for det in results['ensemble_detections']:
            bbox = det.get('bbox')
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color based on risk level
                category = det.get('category', 'SAFE')
                if category == 'CRITICAL':
                    color = (0, 0, 255)  # Red
                elif category == 'DANGER':
                    color = (0, 165, 255)  # Orange
                elif category == 'WARNING':
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                
                # Adjust x coordinate for right side
                x1_r, x2_r = x1 + w, x2 + w
                
                cv2.rectangle(vis, (x1_r, y1), (x2_r, y2), color, 2)
                
                # Label
                label = f"{category}"
                if 'distance_m' in det:
                    label += f" {det['distance_m']:.1f}m"
                
                cv2.putText(vis, label, (x1_r, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add text labels
        cv2.putText(vis, 'Base YOLO', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, 'Ensemble (Multi-Scale + Motion + Depth)', (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add stats
        cv2.putText(vis, f"Base: {results['base_count']} | "
                       f"Motion: {results['motion_count']} | "
                       f"Ensemble: {results['ensemble_count']}",
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Add functional safety grid overlay for better detection guidance
        vis_height, vis_width = vis.shape[:2]

        # Define tractor/equipment position (center-bottom of each side)
        tractor_x_left = w // 2
        tractor_x_right = w + (w // 2)
        tractor_y = int(vis_height * 0.85)

        # Draw concentric safety zones around tractor (on both sides)
        safety_radii = [30, 60, 90, 120, 180]  # Smaller radii for split view
        zone_colors = [
            (0, 255, 0, 0.02),   # Safe zone - very light green
            (0, 255, 255, 0.03), # Low risk - very light yellow
            (0, 165, 255, 0.05), # Medium risk - very light orange
            (0, 0, 255, 0.07),   # High risk - light red
            (0, 0, 139, 0.08)    # Critical zone - light dark red
        ]

        # Draw safety zone circles on left side (Base YOLO)
        for i, (radius, (b, g, r, alpha)) in enumerate(zip(safety_radii, zone_colors)):
            # Create overlay for semi-transparent zones
            overlay = vis.copy()
            cv2.circle(overlay, (tractor_x_left, tractor_y), radius, (b, g, r), -1)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            # Draw zone boundary lines
            cv2.circle(vis, (tractor_x_left, tractor_y), radius, (b, g, r), 1)

        # Draw safety zone circles on right side (Ensemble)
        for i, (radius, (b, g, r, alpha)) in enumerate(zip(safety_radii, zone_colors)):
            # Create overlay for semi-transparent zones
            overlay = vis.copy()
            cv2.circle(overlay, (tractor_x_right, tractor_y), radius, (b, g, r), -1)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            # Draw zone boundary lines
            cv2.circle(vis, (tractor_x_right, tractor_y), radius, (b, g, r), 1)

        # Draw detection quadrants (dividing field of view)
        quadrant_lines = [
            # Vertical center lines for both sides
            ((tractor_x_left, 0), (tractor_x_left, vis_height), (255, 255, 255, 0.05)),
            ((tractor_x_right, 0), (tractor_x_right, vis_height), (255, 255, 255, 0.05)),
            # Horizontal lines at tractor level
            ((0, tractor_y), (vis_width, tractor_y), (255, 255, 255, 0.05)),
        ]

        for start, end, (b, g, r, alpha) in quadrant_lines:
            overlay = vis.copy()
            cv2.line(overlay, start, end, (b, g, r), 1)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # Draw tractor/equipment markers
        cv2.circle(vis, (tractor_x_left, tractor_y), 10, (255, 0, 255), 2)  # Magenta circle left
        cv2.circle(vis, (tractor_x_right, tractor_y), 10, (255, 0, 255), 2)  # Magenta circle right

        # Add zone labels
        cv2.putText(vis, "TRACTOR", (tractor_x_left - 25, tractor_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(vis, "TRACTOR", (tractor_x_right - 25, tractor_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        return vis
    
    def _print_statistics(self):
        """Print processing statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("DEMO PROCESSING COMPLETE - STATISTICS")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 PROCESSING STATISTICS:")
        logger.info(f"   Frames processed: {self.stats['frames_processed']}")
        logger.info(f"   Total detections: {self.stats['total_detections']}")
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            logger.info(f"   Avg processing time: {avg_time*1000:.1f}ms/frame")
            logger.info(f"   Effective FPS: {fps:.1f}")
        
        logger.info(f"\n🚨 RISK LEVELS DETECTED:")
        logger.info(f"   CRITICAL: {self.stats['critical_detections']}")
        logger.info(f"   DANGER: {self.stats['danger_detections']}")
        
        avg_detections = (self.stats['total_detections'] / 
                         max(1, self.stats['frames_processed']))
        logger.info(f"\n📈 DETECTION RATE:")
        logger.info(f"   Avg detections/frame: {avg_detections:.2f}")
        
        logger.info("\n✓ Demo processing finished successfully!")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process demo frames with optimized detection'
    )
    parser.add_argument('--input-type', choices=['webcam', 'video'],
                       default='webcam',
                       help='Input source type')
    parser.add_argument('--input-path', type=str, default='0',
                       help='Path to video file or camera index')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames to process')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable real-time visualization')
    
    args = parser.parse_args()
    
    # Determine input source
    use_webcam = args.input_type == 'webcam'
    input_path = args.input_path if args.input_type == 'video' else None
    
    # Create and run processor
    processor = OptimizedDemoProcessor(
        use_webcam=use_webcam,
        video_path=input_path
    )
    
    processor.run_demo(
        max_frames=args.max_frames,
        show_visualization=not args.no_visualization
    )


if __name__ == '__main__':
    main()
