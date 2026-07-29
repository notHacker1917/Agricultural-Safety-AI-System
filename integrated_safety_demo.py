"""
INTEGRATED AGRICULTURAL SAFETY DEMO

Connects all components end-to-end:
- Camera input → Detections
- Terrain analysis
- Risk assessment (7-factor model)
- Emergency escalation
- Visual monitoring dashboard
- Audit logging

Real-time demonstration of complete safety system on live video feed.
"""

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path
import json
import logging

# Import all safety components
try:
    from tractor_geometry import TractorPOVGeometry, create_realistic_camera, TractorGeometry
    from terrain_analysis import TerrainAnalyzer, TerrainAnalysis
    from context_aware_risk_system import ContextAwareRiskAssessor
    from safety_controller import SafetySystemController, SystemState
    from emergency_protocols import EmergencyResponseController
    from monitoring_dashboard import MonitoringDashboard
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure all modules are in same directory")
    raise

# Configure logging
import os
log_dir = os.path.expanduser("~/safety_logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "integrated_demo.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedSafetyDemo:
    """
    Complete agricultural safety system demonstration.
    Processes video frames through entire safety pipeline.
    """
    
    def __init__(
        self,
        tractor_model: str = "GENERIC",
        max_object_distance: float = 50.0,
        confidence_threshold: float = 0.5,
    ):
        """Initialize all safety components."""
        
        logger.info("=" * 80)
        logger.info("INTEGRATED AGRICULTURAL SAFETY SYSTEM DEMO")
        logger.info("=" * 80)
        
        # Tractor geometry
        self.tractor_geom = TractorPOVGeometry(
            TractorGeometry.default_harvester(tractor_model),
            create_realistic_camera(),
        )
        logger.info(f"Loaded tractor: {tractor_model}")
        
        # Terrain analysis
        self.terrain_analyzer = TerrainAnalyzer(
            max_object_distance=max_object_distance,
        )
        logger.info("Terrain analyzer initialized")
        
        # Risk assessment
        self.risk_assessor = ContextAwareRiskAssessor(
            max_tractor_speed_kmh=5.0,
            reaction_time_seconds=1.0,
        )
        logger.info("Risk assessor initialized")
        
        # Safety controller (main orchestrator)
        self.safety_controller = SafetySystemController(
            tractor_geometry=self.tractor_geom,
            max_tractor_speed_kmh=5.0,
            emergency_stop_latency_ms=100.0,
        )
        logger.info("Safety controller initialized")
        
        # Emergency response controller
        self.emergency_controller = EmergencyResponseController()
        logger.info("Emergency response controller initialized")
        
        # Monitoring dashboard
        self.dashboard = MonitoringDashboard(image_width=1920, image_height=1080)
        logger.info("Monitoring dashboard initialized")
        
        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.max_object_distance = max_object_distance
        
        # Statistics
        self.frames_processed = 0
        self.detections_total = 0
        self.alerts_issued = 0
        self.emergency_stops = 0
        self.start_time = time.time()
        
        logger.info(f"Demo initialized successfully")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Max detection distance: {max_object_distance}m")
        logger.info("=" * 80)
    
    def process_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Process single frame through complete safety pipeline.
        
        Args:
            frame: Input video frame (BGR)
            detections: List of detections from YOLOv8
                       Format: [(x1, y1, x2, y2, confidence, class_id), ...]
        
        Returns:
            Annotated frame with safety overlays
        """
        
        self.frames_processed += 1
        
        # Convert detections to format expected by safety_controller
        detection_dict = {}
        for idx, det in enumerate(detections):
            if len(det) >= 5:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                conf = float(det[4])
                
                # Filter by confidence
                if conf < self.confidence_threshold:
                    continue
                
                detection_dict[idx] = {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": int(det[5]) if len(det) > 5 else 0,
                }
        
        self.detections_total += len(detection_dict)
        
        # Process through safety controller
        try:
            action = self.safety_controller.process_frame(frame, detection_dict)
        except Exception as e:
            logger.error(f"Safety controller error: {e}")
            action = None
        
        # Get current system status
        status = self.safety_controller.get_status_report()
        
        # Analyze terrain (for display)
        terrain = None
        try:
            terrain = self.terrain_analyzer.analyze_image(frame)
        except Exception as e:
            logger.debug(f"Terrain analysis error: {e}")
        
        # Check emergency level
        if action and action.action_type == "EMERGENCY_STOP":
            self.emergency_stops += 1
            emergency_level = 4  # Critical
        elif action and action.action_type == "URGENT_SLOW":
            emergency_level = 2
        elif action and action.action_type == "REDUCE_SPEED":
            emergency_level = 1
        else:
            emergency_level = 0
        
        # Get emergency response
        emergency_response = self.emergency_controller.get_response_for_level(emergency_level)
        
        # Render monitoring dashboard
        output_frame = None
        try:
            output_frame = self.dashboard.render_frame(
                frame,
                detection_dict,
                terrain_analysis=terrain.__dict__ if terrain else None,
                system_status=status,
                current_action=action.to_dict() if action else None,
            )
        except Exception as e:
            logger.error(f"Dashboard rendering error: {e}")
            output_frame = frame.copy()
        
        # Add emergency level indicator
        if emergency_level > 0:
            self._render_emergency_indicator(output_frame, emergency_level, action)
        
        # Add performance stats
        self._render_performance_stats(output_frame)
        
        return output_frame
    
    def _render_emergency_indicator(self, frame: np.ndarray, level: int, action):
        """Render emergency level indicator."""
        h, w = frame.shape[:2]
        
        colors = {
            0: (0, 255, 0),      # Green (safe)
            1: (0, 165, 255),    # Orange (warning)
            2: (0, 100, 255),    # Red-orange (high alert)
            3: (0, 0, 200),      # Dark red (stop imminent)
            4: (0, 0, 255),      # Red (emergency)
            5: (200, 0, 255),    # Magenta (critical failure)
        }
        
        level_names = {
            0: "SAFE",
            1: "WARNING",
            2: "HIGH_ALERT",
            3: "STOP_IMMINENT",
            4: "EMERGENCY_STOP",
            5: "CRITICAL_FAILURE",
        }
        
        color = colors.get(level, (255, 255, 255))
        name = level_names.get(level, "UNKNOWN")
        
        # Large indicator at bottom center
        cv2.rectangle(frame, (w//4, h-80), (3*w//4, h-20), color, -1)
        
        text = f"🚨 {name} LEVEL {level}"
        if action:
            text += f" - {action.action_type}"
        
        cv2.putText(
            frame,
            text,
            (w//4 + 20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
    
    def _render_performance_stats(self, frame: np.ndarray):
        """Render performance statistics."""
        h, w = frame.shape[:2]
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        stats = [
            f"Frames: {self.frames_processed}",
            f"FPS: {fps:.1f}",
            f"Detections: {self.detections_total}",
            f"E-Stops: {self.emergency_stops}",
            f"Time: {elapsed:.1f}s",
        ]
        
        x, y = 10, h - 150
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            y += 30
    
    def shutdown(self):
        """Cleanup and save final report."""
        logger.info("=" * 80)
        logger.info("DEMO SHUTDOWN")
        logger.info("=" * 80)
        
        elapsed = time.time() - self.start_time
        avg_fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        logger.info(f"Total frames processed: {self.frames_processed}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Total detections: {self.detections_total}")
        logger.info(f"Emergency stops: {self.emergency_stops}")
        logger.info(f"Elapsed time: {elapsed:.1f}s")
        
        # Save final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "frames_processed": self.frames_processed,
            "average_fps": avg_fps,
            "total_detections": self.detections_total,
            "emergency_stops": self.emergency_stops,
            "elapsed_seconds": elapsed,
        }
        
        report_path = Path(log_dir) / f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        
        # Shutdown safety controller
        self.safety_controller.shutdown()


def process_webcam_feed(demo: IntegratedSafetyDemo, max_frames: int = None):
    """
    Process live webcam feed through safety system.
    """
    logger.info("Starting webcam feed processing...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frames_read = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break
            
            # Mock detections for demo (replace with real YOLOv8 detections)
            detections = []
            
            # Process frame
            output_frame = demo.process_frame(frame, detections)
            
            # Display
            display_frame = cv2.resize(output_frame, (1280, 720))

            # Add functional safety grid overlay for better detection guidance
            height, width = display_frame.shape[:2]

            # Define tractor/equipment position (center-bottom of frame)
            tractor_x, tractor_y = int(width * 0.5), int(height * 0.85)

            # Draw 3x3 grid lines (4 vertical, 4 horizontal lines creating 9 equal squares)
            grid_color = (255, 255, 255)  # White
            grid_alpha = 0.02  # Very light overlay

            # Vertical grid lines (every 1/3 of width)
            for i in range(1, 3):  # 2 lines creating 3 columns
                x = int(width * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Horizontal grid lines (every 1/3 of height)
            for i in range(1, 3):  # 2 lines creating 3 rows
                y = int(height * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (0, y), (width, y), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Draw tractor/equipment marker
            cv2.circle(display_frame, (tractor_x, tractor_y), 12, (255, 0, 255), 2)  # Smaller, thinner magenta circle
            cv2.putText(display_frame, "TRACTOR", (tractor_x - 25, tractor_y - 15),
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
                cv2.putText(display_frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            cv2.imshow("Agricultural Safety System - Live Demo", display_frame)
            
            frames_read += 1
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
            
            if max_frames and frames_read >= max_frames:
                logger.info(f"Reached max frames ({max_frames})")
                break
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        demo.shutdown()


def process_video_file(demo: IntegratedSafetyDemo, video_path: str, max_frames: int = None):
    """
    Process video file through safety system.
    """
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    frames_read = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video")
                break
            
            # Mock detections for demo
            detections = []
            
            # Process frame
            output_frame = demo.process_frame(frame, detections)
            
            # Display
            display_frame = cv2.resize(output_frame, (1280, 720))

            # Add functional safety grid overlay for better detection guidance
            height, width = display_frame.shape[:2]

            # Define tractor/equipment position (center-bottom of frame)
            tractor_x, tractor_y = int(width * 0.5), int(height * 0.85)

            # Draw 3x3 grid lines (4 vertical, 4 horizontal lines creating 9 equal squares)
            grid_color = (255, 255, 255)  # White
            grid_alpha = 0.02  # Very light overlay

            # Vertical grid lines (every 1/3 of width)
            for i in range(1, 3):  # 2 lines creating 3 columns
                x = int(width * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (x, 0), (x, height), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Horizontal grid lines (every 1/3 of height)
            for i in range(1, 3):  # 2 lines creating 3 rows
                y = int(height * i / 3)
                overlay = display_frame.copy()
                cv2.line(overlay, (0, y), (width, y), grid_color, 1)
                cv2.addWeighted(overlay, grid_alpha, display_frame, 1 - grid_alpha, 0, display_frame)

            # Draw tractor/equipment marker
            cv2.circle(display_frame, (tractor_x, tractor_y), 12, (255, 0, 255), 2)  # Smaller, thinner magenta circle
            cv2.putText(display_frame, "TRACTOR", (tractor_x - 25, tractor_y - 15),
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
                cv2.putText(display_frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            cv2.imshow("Agricultural Safety System - Video Demo", display_frame)
            
            frames_read += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
            
            if max_frames and frames_read >= max_frames:
                logger.info(f"Reached max frames ({max_frames})")
                break
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        demo.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Agricultural Safety System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live webcam demo
  python integrated_safety_demo.py --mode webcam
  
  # Process video file
  python integrated_safety_demo.py --mode video --input-path video.mp4
  
  # Run for specific number of frames
  python integrated_safety_demo.py --mode webcam --max-frames 300
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["webcam", "video"],
        default="webcam",
        help="Input mode (default: webcam)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="0",
        help="Video file path (for video mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--tractor",
        type=str,
        default="GENERIC",
        choices=["GENERIC", "CLAAS", "JOHNDEERE", "MASSEY"],
        help="Tractor model",
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = IntegratedSafetyDemo(
        tractor_model=args.tractor,
        confidence_threshold=args.confidence,
    )
    
    # Process input
    if args.mode == "webcam":
        process_webcam_feed(demo, max_frames=args.max_frames)
    else:
        process_video_file(demo, args.input_path, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
