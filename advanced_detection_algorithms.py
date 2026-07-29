"""
ADVANCED HUMAN DETECTION ALGORITHMS FOR AGRICULTURE
Multi-scale, Motion-aware, and Contextual Detection Methods
Designed to work with demo frames and real-time processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. MULTI-SCALE HUMAN DETECTOR
# ============================================================================

class MultiScaleHumanDetector:
    """Detect humans at multiple scales for far and near objects"""
    
    def __init__(self, base_detector, scales: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0]):
        """
        Args:
            base_detector: YOLO detector or similar
            scales: Pyramid scales to process
        """
        self.base_detector = base_detector
        self.scales = scales
        self.detections_cache = {}
        
    def detect_multi_scale(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect humans at multiple image scales
        
        Returns:
            List of detections with scale metadata
        """
        h, w = frame.shape[:2]
        all_detections = []
        scale_detections = {}
        
        for scale in self.scales:
            # Resize frame
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            
            if scaled_h < 32 or scaled_w < 32:
                continue
            
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Detect at this scale
            try:
                detections = self.base_detector.detect(scaled_frame)
            except:
                continue
            
            # Un-scale bounding boxes
            for bbox, conf in detections:
                x1, y1, x2, y2 = bbox
                original_bbox = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale)
                )
                
                detection_entry = {
                    'bbox': original_bbox,
                    'confidence': conf,
                    'scale': scale,
                    'metadata': {'scale_factor': scale}
                }
                all_detections.append(detection_entry)
                
                # Track by scale
                if scale not in scale_detections:
                    scale_detections[scale] = []
                scale_detections[scale].append(detection_entry)
        
        # De-duplicate detections using NMS across scales
        logger.info(f"Multi-scale detection: {sum(len(d) for d in scale_detections.values())} detections before NMS")
        filtered_detections = self._nms_multi_scale(all_detections, frame.shape[:2])
        logger.info(f"After NMS: {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def _nms_multi_scale(self, detections: List[Dict], frame_shape: Tuple, 
                         iou_threshold: float = 0.5) -> List[Dict]:
        """Non-maximum suppression across scales"""
        if not detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        removed = set()
        
        for i, det in enumerate(sorted_detections):
            if i in removed:
                continue
            
            kept.append(det)
            
            # Compare with remaining detections
            for j in range(i + 1, len(sorted_detections)):
                if j in removed:
                    continue
                
                iou = self._calculate_iou(det['bbox'], sorted_detections[j]['bbox'])
                if iou > iou_threshold:
                    removed.add(j)
        
        return kept
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0


# ============================================================================
# 2. MOTION-BASED HUMAN DETECTOR
# ============================================================================

class MotionBasedDetector:
    """Detect humans using motion/optical flow analysis"""
    
    def __init__(self, motion_threshold: float = 10.0, min_area: int = 100):
        """
        Args:
            motion_threshold: Pixels per frame threshold (increased for less sensitivity)
            min_area: Minimum motion blob area (increased for larger objects only)
        """
        self.prev_gray = None
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        
    def detect_motion(self, frame: np.ndarray, 
                     exclude_static_regions: bool = True) -> List[Dict]:
        """
        Detect moving objects using optical flow
        
        Returns:
            List of motion detections with bounding boxes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        
        h, w = gray.shape
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate motion magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Thresholding: motion above threshold
        motion_mask = magnitude > self.motion_threshold
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), 
                                       cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (human-like) - more restrictive
            aspect_ratio = w_box / max(h_box, 1)
            if aspect_ratio > 1.8 or aspect_ratio < 0.4:  # More restrictive range
                continue
            
            # Additional size filtering - humans should be reasonably sized
            if w_box < 20 or h_box < 40:  # Minimum human size
                continue
            
            detections.append({
                'bbox': (x, y, x + w_box, y + h_box),
                'confidence': 0.7,  # Motion confidence
                'type': 'motion',
                'area': area
            })
        
        self.prev_gray = gray
        return detections


# ============================================================================
# 3. DEPTH-BASED HUMAN DETECTOR
# ============================================================================

class DepthBasedDetector:
    """Estimate depth and categorize humans by distance"""
    
    def __init__(self, ref_height_pixel: float = 50.0, ref_distance_m: float = 10.0):
        """
        Args:
            ref_height_pixel: Reference height in pixels (e.g., person at 10m = 50px)
            ref_distance_m: Reference distance in meters
        """
        self.ref_height_pixel = ref_height_pixel
        self.ref_distance_m = ref_distance_m
        
    def estimate_distance(self, bbox: Tuple, frame_h: int) -> float:
        """
        Estimate distance to human based on bbox height
        
        Formula: distance = ref_distance * (ref_height / bbox_height)
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        if bbox_height <= 0:
            return float('inf')
        
        distance = self.ref_distance_m * (self.ref_height_pixel / bbox_height)
        return distance
    
    def categorize_detections(self, detections: List[Dict], 
                             frame_h: int) -> List[Dict]:
        """
        Categorize detections by estimated distance
        
        Categories:
            - CRITICAL: < 5m
            - DANGER: 5-15m
            - WARNING: 15-30m
            - SAFE: > 30m
        """
        categorized = []
        
        for det in detections:
            bbox = det['bbox']
            distance = self.estimate_distance(bbox, frame_h)
            
            if distance < 5:
                category = 'CRITICAL'
                risk_score = 1.0
            elif distance < 15:
                category = 'DANGER'
                risk_score = 0.7
            elif distance < 30:
                category = 'WARNING'
                risk_score = 0.4
            else:
                category = 'SAFE'
                risk_score = 0.1
            
            det['distance_m'] = distance
            det['category'] = category
            det['risk_score'] = risk_score
            
            categorized.append(det)
        
        return categorized


# ============================================================================
# 4. CONTEXTUAL AWARENESS DETECTOR
# ============================================================================

class ContextualAwarenessDetector:
    """Add contextual information to detections"""
    
    def __init__(self, frame_history_size: int = 5):
        """
        Args:
            frame_history_size: Number of frames to keep for context
        """
        self.frame_history = []
        self.detection_history = []
        self.frame_history_size = frame_history_size
        
    def add_context(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Add contextual information to detections
        """
        # Store frame and detections
        self.frame_history.append(frame.copy())
        self.detection_history.append(detections)
        
        # Keep history limited
        if len(self.frame_history) > self.frame_history_size:
            self.frame_history.pop(0)
            self.detection_history.pop(0)
        
        # Analyze context
        contextualized = []
        for i, det in enumerate(detections):
            det['temporal_consistency'] = self._calculate_consistency(det, i)
            det['background_complexity'] = self._estimate_background_complexity(
                frame, det['bbox']
            )
            
            contextualized.append(det)
        
        return contextualized
    
    def _calculate_consistency(self, current_det: Dict, det_idx: int) -> float:
        """
        Calculate consistency across recent frames
        (how stable is this detection across frames)
        """
        if len(self.detection_history) < 2:
            return 1.0
        
        # Check if similar detection exists in previous frame
        current_bbox = current_det['bbox']
        consistency_score = 0.0
        found_matches = 0
        
        for prev_detections in self.detection_history[:-1]:
            if det_idx < len(prev_detections):
                prev_bbox = prev_detections[det_idx]['bbox']
                
                # Calculate IoU
                iou = self._calculate_iou(current_bbox, prev_bbox)
                if iou > 0.3:
                    consistency_score += iou
                    found_matches += 1
        
        if found_matches > 0:
            consistency_score /= found_matches
        
        return consistency_score
    
    def _estimate_background_complexity(self, frame: np.ndarray, 
                                       bbox: Tuple) -> float:
        """
        Estimate background complexity in detection region
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.5
        
        # Calculate Laplacian variance (edge complexity)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize (typically 0-1000)
        complexity = min(1.0, laplacian_var / 1000.0)
        
        return complexity
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        intersection_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersection_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = intersection_x * intersection_y
        
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# ============================================================================
# 5. PREPROCESSING FOR ROBUST DETECTION
# ============================================================================

class RobustPreprocessor:
    """Adaptive preprocessing for challenging lighting/contrast conditions"""
    
    @staticmethod
    def adaptive_preprocessing(frame: np.ndarray, 
                             target_brightness: int = 127) -> np.ndarray:
        """
        Apply adaptive preprocessing for robustness
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive histogram equalization on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Check brightness and adjust if needed
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            # Too dark - increase brightness
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=20)
        elif brightness > 200:
            # Too bright - reduce brightness
            processed = cv2.convertScaleAbs(processed, alpha=0.9, beta=-10)
        
        return processed
    
    @staticmethod
    def suppress_reflections(frame: np.ndarray, threshold: int = 250) -> np.ndarray:
        """
        Suppress bright reflections that can cause false detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > threshold
        
        if np.sum(bright_mask) > 0:
            # Replace bright pixels with neighboring values
            frame = frame.copy()
            inpainted = cv2.inpaint(frame, bright_mask.astype(np.uint8) * 255, 3,
                                   cv2.INPAINT_TELEA)
            return inpainted
        
        return frame


# ============================================================================
# 6. ENSEMBLE HUMAN DETECTOR
# ============================================================================

class EnsembleHumanDetector:
    """Combine multiple detection methods for robust results"""
    
    def __init__(self, base_yolo_detector, use_motion: bool = True,
                 use_depth: bool = True, use_context: bool = True):
        """
        Args:
            base_yolo_detector: Main YOLO detector
            use_motion: Enable motion-based detection
            use_depth: Enable depth analysis
            use_context: Enable contextual awareness
        """
        self.base_detector = base_yolo_detector
        self.multi_scale = MultiScaleHumanDetector(base_yolo_detector)
        self.motion_detector = MotionBasedDetector() if use_motion else None
        self.depth_detector = DepthBasedDetector() if use_depth else None
        self.context_detector = ContextualAwarenessDetector() if use_context else None
        self.preprocessor = RobustPreprocessor()
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Ensemble detection combining multiple methods
        
        Returns:
            List of robust detections with high confidence
        """
        # Preprocess for robustness
        processed_frame = self.preprocessor.adaptive_preprocessing(frame)
        
        all_detections = []
        detection_counts = {}
        
        # 1. Multi-scale YOLO detection
        try:
            yolo_detections = self.multi_scale.detect_multi_scale(processed_frame)
            logger.info(f"  ✓ YOLO: {len(yolo_detections)} detections")
            
            for det in yolo_detections:
                det['method'] = 'YOLO'
                all_detections.append(det)
                bbox_key = self._bbox_to_key(det['bbox'])
                detection_counts[bbox_key] = detection_counts.get(bbox_key, 0) + 1
        except Exception as e:
            logger.warning(f"  ✗ YOLO detection failed: {e}")
        
        # 2. Motion-based detection
        if self.motion_detector:
            try:
                motion_detections = self.motion_detector.detect_motion(processed_frame)
                logger.info(f"  ✓ Motion: {len(motion_detections)} detections")
                
                for det in motion_detections:
                    det['method'] = 'Motion'
                    all_detections.append(det)
                    bbox_key = self._bbox_to_key(det['bbox'])
                    detection_counts[bbox_key] = detection_counts.get(bbox_key, 0) + 1
            except Exception as e:
                logger.warning(f"  ✗ Motion detection failed: {e}")
        
        # 3. Add depth information
        if self.depth_detector:
            all_detections = self.depth_detector.categorize_detections(
                all_detections, frame.shape[0]
            )
        
        # 4. Add contextual information
        if self.context_detector:
            all_detections = self.context_detector.add_context(all_detections, frame)
        
        # 5. Filter and aggregate detections
        filtered_detections = self._filter_ensemble_detections(
            all_detections, detection_counts
        )
        
        logger.info(f"  ═ Ensemble result: {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def _bbox_to_key(self, bbox: Tuple, grid_size: int = 20) -> Tuple:
        """Create a key for bbox clustering"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (int(cx // grid_size), int(cy // grid_size))
    
    def _filter_ensemble_detections(self, detections: List[Dict],
                                   detection_counts: Dict,
                                   min_confidence: float = 0.6) -> List[Dict]:
        """
        Filter detections based on confidence and ensemble voting
        Prioritize YOLO detections over motion-only detections
        """
        filtered = []
        seen_regions = set()
        
        # Separate YOLO and motion detections
        yolo_detections = [d for d in detections if d.get('method') == 'YOLO']
        motion_detections = [d for d in detections if d.get('method') == 'Motion']
        
        # Process YOLO detections first (higher priority)
        for det in sorted(yolo_detections, key=lambda x: x.get('confidence', 0.5), reverse=True):
            if det.get('confidence', 0) < min_confidence:
                continue
                
            bbox_key = self._bbox_to_key(det['bbox'])
            if bbox_key in seen_regions:
                continue
                
            ensemble_votes = detection_counts.get(bbox_key, 1)
            det['ensemble_votes'] = ensemble_votes
            det['ensemble_confidence'] = min(1.0, det.get('confidence', 0.5) * 1.1)  # Boost YOLO confidence
            
            filtered.append(det)
            seen_regions.add(bbox_key)
        
        # Add motion detections only if they don't conflict with YOLO and have high confidence
        for det in sorted(motion_detections, key=lambda x: x.get('confidence', 0.5), reverse=True):
            if det.get('confidence', 0) < 0.8:  # Higher threshold for motion-only
                continue
                
            bbox_key = self._bbox_to_key(det['bbox'])
            if bbox_key in seen_regions:
                continue
                
            # Only add motion detection if it's in a region not covered by YOLO
            ensemble_votes = detection_counts.get(bbox_key, 1)
            if ensemble_votes >= 2:  # Only if motion + something else agrees
                det['ensemble_votes'] = ensemble_votes
                det['ensemble_confidence'] = det.get('confidence', 0.5)
                filtered.append(det)
                seen_regions.add(bbox_key)
        
        return filtered


def test_ensemble_detector():
    """Test the ensemble detector"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ENSEMBLE HUMAN DETECTOR")
    logger.info("=" * 80)
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test preprocessing
    preprocessor = RobustPreprocessor()
    processed = preprocessor.adaptive_preprocessing(test_frame)
    logger.info("✓ Preprocessing works")
    
    # Test motion detector
    motion_detector = MotionBasedDetector()
    motion_dets = motion_detector.detect_motion(test_frame)
    logger.info(f"✓ Motion detector initialized: {len(motion_dets)} detections on test frame")
    
    # Test depth detector
    depth_detector = DepthBasedDetector()
    test_det = {'bbox': (100, 50, 200, 200)}
    distance = depth_detector.estimate_distance(test_det['bbox'], 480)
    logger.info(f"✓ Depth detector: estimated distance = {distance:.1f}m")
    
    # Test contextual awareness
    context_detector = ContextualAwarenessDetector()
    contextualized = context_detector.add_context([test_det], test_frame)
    logger.info(f"✓ Contextual detector: {len(contextualized)} detections")
    
    logger.info("\n✓ All components initialized successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    test_ensemble_detector()
