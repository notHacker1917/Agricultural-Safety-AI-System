"""
Upgraded Detection Module for Agricultural Safety AI System
Implements:
- Multi-scale inference with confidence-aware NMS
- SAHI slicing for small/far human detection
- Agricultural preprocessing (CLAHE, brightness normalization, shadow suppression)
- Detection filtering with human aspect ratio validation
- Confidence fusion (YOLO + motion + temporal)
- Temporal detection stability
- Detection evaluation hooks
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ultralytics import YOLO
import torch

from config import get_config, DetectionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Standardized detection data structure."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # 0 = person
    track_id: Optional[int] = None
    scale: float = 1.0
    is_occluded: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    temporal_confidence: float = 0.0
    motion_confidence: float = 0.0
    fused_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def aspect_ratio(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return w / h if h > 0 else 0.0


class AgriculturalPreprocessor:
    """Enhanced preprocessing for agricultural environments."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size)
        )
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline."""
        processed = frame.copy()
        
        # 1. Brightness normalization
        processed = self._normalize_brightness(processed)
        
        # 2. Shadow suppression
        processed = self._suppress_shadows(processed)
        
        # 3. CLAHE contrast enhancement
        processed = self._apply_clahe(processed)
        
        # 4. Dust/noise reduction
        processed = self._reduce_noise(processed)
        
        return processed
    
    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Normalize brightness using LAB color space."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for brightness normalization
        l = self.clahe.apply(l)
        
        # Adjust overall brightness if needed
        mean_brightness = np.mean(l)
        if mean_brightness < self.config.brightness_target - 30:
            # Too dark - brighten
            l = cv2.addWeighted(l, 1.2, np.zeros_like(l), 0, 20)
        elif mean_brightness > self.config.brightness_target + 30:
            # Too bright - darken
            l = cv2.addWeighted(l, 0.9, np.zeros_like(l), 0, -10)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _suppress_shadows(self, image: np.ndarray) -> np.ndarray:
        """Suppress shadows using morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create shadow mask (dark regions)
        _, shadow_mask = cv2.threshold(gray, self.config.shadow_suppression_threshold, 255, cv2.THRESH_BINARY_INV)
        
        if np.sum(shadow_mask) > 0:
            # Dilate shadow mask to cover shadow boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=2)
            
            # Inpaint shadow regions
            image = cv2.inpaint(image, shadow_mask, 3, cv2.INPAINT_TELEA)
        
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce dust and noise using bilateral filtering."""
        return cv2.bilateralFilter(
            image,
            d=self.config.dust_noise_kernel_size,
            sigmaColor=75,
            sigmaSpace=75
        )


class MultiScaleDetector:
    """Multi-scale inference with confidence-aware NMS."""
    
    def __init__(self, model: YOLO, config: DetectionConfig):
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run multi-scale detection and merge results."""
        all_detections = []
        
        for scale in self.config.inference_scales:
            # Resize frame to target scale
            h, w = frame.shape[:2]
            target_w, target_h = scale, scale
            
            # Maintain aspect ratio with padding
            scale_factor = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            padded = cv2.copyMakeBorder(
                resized, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            
            # Run inference
            try:
                results = self.model(
                    padded,
                    classes=[0],  # Person class only
                    conf=self.config.base_confidence,
                    device=self.device,
                    half=self.device == 'cuda',
                    verbose=False
                )
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get bbox and scale back to original frame
                            bbox = box.xyxy[0].cpu().numpy()
                            x1 = (bbox[0] - pad_w) / scale_factor
                            y1 = (bbox[1] - pad_h) / scale_factor
                            x2 = (bbox[2] - pad_w) / scale_factor
                            y2 = (bbox[3] - pad_h) / scale_factor
                            
                            # Clamp to frame bounds
                            x1 = max(0, min(int(x1), w))
                            y1 = max(0, min(int(y1), h))
                            x2 = max(0, min(int(x2), w))
                            y2 = max(0, min(int(y2), h))
                            
                            conf = float(box.conf[0].cpu().numpy())
                            
                            detection = Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                scale=scale
                            )
                            all_detections.append(detection)
                            
            except Exception as e:
                logger.warning(f"Multi-scale detection failed at scale {scale}: {e}")
                continue
        
        # Merge detections using confidence-aware NMS
        merged = self._confidence_aware_nms(all_detections)
        return merged
    
    def _confidence_aware_nms(self, detections: List[Detection]) -> List[Detection]:
        """NMS that considers confidence and scale."""
        if not detections:
            return []
        
        # Sort by confidence (higher first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        kept = []
        removed_indices = set()
        
        for i, det_i in enumerate(detections):
            if i in removed_indices:
                continue
            
            kept.append(det_i)
            
            for j in range(i + 1, len(detections)):
                if j in removed_indices:
                    continue
                
                det_j = detections[j]
                iou = self._calculate_iou(det_i.bbox, det_j.bbox)
                
                if iou > self.config.scale_merge_iou_threshold:
                    # Keep the one with higher confidence
                    removed_indices.add(j)
        
        return kept
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class SAHIDetector:
    """SAHI-based slicing for small/far human detection."""
    
    def __init__(self, model: YOLO, config: DetectionConfig):
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run sliced inference for small object detection."""
        if not self.config.sahi_enabled:
            return []
        
        h, w = frame.shape[:2]
        all_detections = []
        
        slice_h = self.config.sahi_slice_height
        slice_w = self.config.sahi_slice_width
        overlap_h = int(slice_h * self.config.sahi_overlap_height_ratio)
        overlap_w = int(slice_w * self.config.sahi_overlap_width_ratio)
        
        # Generate slices
        y_offsets = list(range(0, h - slice_h + 1, slice_h - overlap_h))
        x_offsets = list(range(0, w - slice_w + 1, slice_w - overlap_w))
        
        # Ensure we cover the edges
        if y_offsets[-1] + slice_h < h:
            y_offsets.append(h - slice_h)
        if x_offsets[-1] + slice_w < w:
            x_offsets.append(w - slice_w)
        
        for y_offset in y_offsets:
            for x_offset in x_offsets:
                # Extract slice
                slice_img = frame[y_offset:y_offset + slice_h, x_offset:x_offset + slice_w]
                
                try:
                    results = self.model(
                        slice_img,
                        classes=[0],
                        conf=self.config.base_confidence * 0.8,  # Lower threshold for small objects
                        device=self.device,
                        half=self.device == 'cuda',
                        verbose=False
                    )
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                bbox = box.xyxy[0].cpu().numpy()
                                # Map back to original frame coordinates
                                x1 = int(bbox[0] + x_offset)
                                y1 = int(bbox[1] + y_offset)
                                x2 = int(bbox[2] + x_offset)
                                y2 = int(bbox[3] + y_offset)
                                
                                # Clamp to frame bounds
                                x1 = max(0, min(x1, w))
                                y1 = max(0, min(y1, h))
                                x2 = max(0, min(x2, w))
                                y2 = max(0, min(y2, h))
                                
                                conf = float(box.conf[0].cpu().numpy())
                                
                                detection = Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=conf,
                                    metadata={'source': 'sahi'}
                                )
                                all_detections.append(detection)
                                
                except Exception as e:
                    logger.debug(f"SAHI slice detection failed at ({x_offset}, {y_offset}): {e}")
                    continue
        
        # Merge overlapping detections
        merged = self._merge_overlapping(all_detections)
        return merged
    
    def _merge_overlapping(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections from slices."""
        if not detections:
            return []
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []
        removed = set()
        
        for i, det_i in enumerate(detections):
            if i in removed:
                continue
            
            kept.append(det_i)
            
            for j in range(i + 1, len(detections)):
                if j in removed:
                    continue
                
                iou = self._calculate_iou(det_i.bbox, detections[j].bbox)
                if iou > 0.7:  # High overlap threshold for merging
                    removed.add(j)
        
        return kept
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class DetectionFilter:
    """Filter detections based on human properties and frame position."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def filter(self, detections: List[Detection], frame_shape: Tuple) -> List[Detection]:
        """Apply all filtering rules."""
        h, w = frame_shape[:2]
        frame_area = h * w
        filtered = []
        
        for det in detections:
            # 1. Aspect ratio validation (human-like proportions)
            if not (self.config.min_aspect_ratio <= det.aspect_ratio <= self.config.max_aspect_ratio):
                continue
            
            # 2. Minimum bbox size threshold
            area_ratio = det.area / frame_area
            if area_ratio < self.config.min_bbox_area_ratio:
                continue
            
            # 3. Maximum bbox size threshold
            if area_ratio > self.config.max_bbox_area_ratio:
                continue
            
            # 4. Edge-of-frame filtering
            margin_x = int(w * self.config.edge_margin_ratio)
            margin_y = int(h * self.config.edge_margin_ratio)
            if (det.bbox[0] < margin_x or det.bbox[1] < margin_y or
                det.bbox[2] > w - margin_x or det.bbox[3] > h - margin_y):
                continue
            
            filtered.append(det)
        
        return filtered


class TemporalStabilityTracker:
    """Track detection stability across frames for temporal confidence."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        # Track detection history by position
        self.position_history = defaultdict(lambda: deque(maxlen=config.detection_persistence_frames))
        self.confidence_history = defaultdict(lambda: deque(maxlen=30))  # Longer history for smoothing
        self.frame_counter = 0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update temporal confidence for detections."""
        self.frame_counter += 1
        current_positions = set()
        
        for det in detections:
            # Create position key (quantized)
            cx, cy = det.center
            pos_key = (int(cx) // 20, int(cy) // 20)  # 20-pixel grid
            current_positions.add(pos_key)
            
            # Update position history
            self.position_history[pos_key].append(self.frame_counter)
            
            # Update confidence history
            self.confidence_history[pos_key].append(det.confidence)
            
            # Calculate temporal confidence
            # How many recent frames had a detection at this position?
            recent_frames = [f for f in self.position_history[pos_key] 
                          if f > self.frame_counter - self.config.detection_persistence_frames]
            temporal_conf = len(recent_frames) / self.config.detection_persistence_frames
            
            det.temporal_confidence = temporal_conf
        
        # Clean up old position histories
        self._cleanup_old_positions()
        
        return detections
    
    def _cleanup_old_positions(self):
        """Remove position histories that haven't been updated recently."""
        keys_to_remove = []
        for pos_key, history in self.position_history.items():
            if history and max(history) < self.frame_counter - self.config.detection_persistence_frames * 2:
                keys_to_remove.append(pos_key)
        
        for key in keys_to_remove:
            del self.position_history[key]
            if key in self.confidence_history:
                del self.confidence_history[key]


class ConfidenceFuser:
    """Fuse YOLO confidence with motion and temporal confidence."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        # Weights for fusion
        self.yolo_weight = 0.5
        self.motion_weight = 0.2
        self.temporal_weight = 0.3
    
    def fuse(self, detections: List[Detection]) -> List[Detection]:
        """Calculate fused confidence for each detection."""
        for det in detections:
            # Weighted combination
            yolo_conf = det.confidence
            motion_conf = det.motion_confidence
            temporal_conf = det.temporal_confidence
            
            fused = (
                self.yolo_weight * yolo_conf +
                self.motion_weight * motion_conf +
                self.temporal_weight * temporal_conf
            )
            
            det.fused_confidence = min(1.0, max(0.0, fused))
        
        return detections


class DetectionEvaluator:
    """Evaluation hooks for detection metrics."""
    
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'total_detections': 0,
            'filtered_detections': 0,
            'small_object_detections': 0,
            'false_negatives': 0,
            'precision_sum': 0.0,
            'recall_sum': 0.0,
            'evaluation_frames': 0
        }
        self.detection_sizes = []
    
    def log_detections(self, detections: List[Detection], frame_shape: Tuple):
        """Log detection statistics."""
        h, w = frame_shape[:2]
        frame_area = h * w
        
        self.metrics['total_detections'] += len(detections)
        
        for det in detections:
            area_ratio = det.area / frame_area
            
            # Track small object detections
            if area_ratio < 0.01:  # Less than 1% of frame
                self.metrics['small_object_detections'] += 1
            
            self.detection_sizes.append(det.area)
        
        # Keep size history manageable
        if len(self.detection_sizes) > 1000:
            self.detection_sizes = self.detection_sizes[-1000:]
    
    def evaluate_frame(self, detections: List[Detection], 
                      previous_detections: List[Detection]) -> Dict:
        """Evaluate detection quality for a frame."""
        if not self.config.evaluation.enable_evaluation:
            return {}
        
        self.metrics['evaluation_frames'] += 1
        
        # Simple precision estimation based on detection consistency
        if previous_detections:
            consistent = 0
            for det in detections:
                for prev_det in previous_detections:
                    iou = self._calculate_iou(det.bbox, prev_det.bbox)
                    if iou > self.config.evaluation.precision_threshold:
                        consistent += 1
                        break
            
            precision = consistent / max(len(detections), 1)
            self.metrics['precision_sum'] += precision
        
        # Log metrics periodically
        if self.metrics['evaluation_frames'] % self.config.evaluation.evaluation_interval == 0:
            return self._get_metrics_summary()
        
        return {}
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_metrics_summary(self) -> Dict:
        """Get summary of detection metrics."""
        avg_size = np.mean(self.detection_sizes) if self.detection_sizes else 0
        precision = (self.metrics['precision_sum'] / 
                    max(self.metrics['evaluation_frames'], 1))
        
        return {
            'avg_detection_size': avg_size,
            'precision_estimate': precision,
            'total_detections': self.metrics['total_detections'],
            'small_object_detections': self.metrics['small_object_detections'],
            'evaluation_frames': self.metrics['evaluation_frames']
        }
    
    def reset(self):
        """Reset metrics."""
        self.metrics = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.metrics.items()}
        self.detection_sizes.clear()


class UpgradedObjectDetector:
    """
    Upgraded object detector with all improvements.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', config: Optional[Any] = None):
        """
        Initialize upgraded detector.
        
        Args:
            model_path: Path to YOLO model
            config: Optional configuration override (can be full SystemConfig or DetectionConfig)
        """
        # Handle different config types
        if config is None:
            full_config = get_config()
            self.config = full_config.detection
            self.full_config = full_config
        elif hasattr(config, 'detection'):
            # It's a SystemConfig
            self.config = config.detection
            self.full_config = config
        else:
            # It's already a DetectionConfig
            self.config = config
            self.full_config = get_config()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing upgraded detector on device: {self.device}")
        
        # Load model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
        
        # Initialize components
        self.preprocessor = AgriculturalPreprocessor(self.config)
        self.multi_scale_detector = MultiScaleDetector(self.model, self.config) if self.model else None
        self.sahi_detector = SAHIDetector(self.model, self.config) if self.model else None
        self.filter = DetectionFilter(self.config)
        self.temporal_tracker = TemporalStabilityTracker(self.config)
        self.confidence_fuser = ConfidenceFuser(self.config)
        self.evaluator = DetectionEvaluator(self.full_config)
        
        # Motion detector for motion confidence
        self.prev_gray = None
        self.motion_detector_enabled = True
        
        # State
        self.previous_detections = []
        self.frame_count = 0
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run full detection pipeline.
        
        Returns:
            List of detection dictionaries with all metadata
        """
        self.frame_count += 1
        
        if self.model is None:
            logger.warning("Model not loaded, returning empty detections")
            return []
        
        # 1. Preprocess frame
        processed_frame = self.preprocessor.preprocess(frame)
        
        # 2. Run multi-scale detection
        detections = self.multi_scale_detector.detect(processed_frame)
        
        # 3. Run SAHI detection for small objects
        sahi_detections = self.sahi_detector.detect(processed_frame)
        
        # 4. Merge detections (SAHI adds to multi-scale)
        detections.extend(sahi_detections)
        
        # 5. Calculate motion confidence
        detections = self._calculate_motion_confidence(processed_frame, detections)
        
        # 6. Filter detections
        detections = self.filter.filter(detections, frame.shape)
        
        # 7. Update temporal stability
        detections = self.temporal_tracker.update(detections)
        
        # 8. Fuse confidence scores
        detections = self.confidence_fuser.fuse(detections)
        
        # 9. Filter by fused confidence threshold
        detections = [d for d in detections if d.fused_confidence >= self.config.base_confidence * 0.8]
        
        # 10. Log and evaluate
        self.evaluator.log_detections(detections, frame.shape)
        eval_metrics = self.evaluator.evaluate_frame(detections, self.previous_detections)
        
        # Store for next frame
        self.previous_detections = detections
        
        # Convert to dictionary format for compatibility
        result = []
        for det in detections:
            det_dict = {
                'bbox': det.bbox,
                'confidence': det.fused_confidence,
                'yolo_confidence': det.confidence,
                'temporal_confidence': det.temporal_confidence,
                'motion_confidence': det.motion_confidence,
                'center': det.center,
                'area': det.area,
                'aspect_ratio': det.aspect_ratio,
                'scale': det.scale,
                'is_occluded': det.is_occluded,
                'velocity': det.velocity,
                'metadata': det.metadata
            }
            if eval_metrics:
                det_dict['eval_metrics'] = eval_metrics
            result.append(det_dict)
        
        return result
    
    def _calculate_motion_confidence(self, frame: np.ndarray, 
                                    detections: List[Detection]) -> List[Detection]:
        """Calculate motion confidence based on optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None and self.motion_detector_enabled:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                # Get average motion magnitude in detection region
                region_magnitude = magnitude[y1:y2, x1:x2]
                if region_magnitude.size > 0:
                    avg_magnitude = np.mean(region_magnitude)
                    # Normalize to 0-1 range (assuming max motion ~20 pixels/frame)
                    det.motion_confidence = min(1.0, avg_magnitude / 20.0)
        
        self.prev_gray = gray
        return detections
    
    def get_evaluation_metrics(self) -> Dict:
        """Get current evaluation metrics."""
        return self.evaluator._get_metrics_summary()
    
    def reset(self):
        """Reset detector state."""
        self.prev_gray = None
        self.previous_detections = []
        self.frame_count = 0
        self.evaluator.reset()


# Backward compatibility alias
ObjectDetector = UpgradedObjectDetector


def test_upgraded_detector():
    """Test the upgraded detector."""
    logger.info("Testing Upgraded Object Detector")
    
    config = get_config()
    detector = UpgradedObjectDetector('yolov8n.pt', config)
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(test_frame)
    logger.info(f"Detected {len(detections)} objects")
    
    # Get metrics
    metrics = detector.get_evaluation_metrics()
    logger.info(f"Evaluation metrics: {metrics}")
    
    logger.info("Upgraded detector test complete")


if __name__ == '__main__':
    test_upgraded_detector()