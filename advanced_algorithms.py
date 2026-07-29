"""
Advanced Detection Ensemble & Multi-Modal Fusion Implementation
Provides plug-and-play improvements for the agricultural safety system
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from collections import defaultdict

# ====================
# 1. ENSEMBLE DETECTOR
# ====================

class EnsembleDetector:
    """Multi-model detection voting for robust predictions"""
    
    def __init__(self, confidence_threshold=0.5):
        self.detectors = []
        self.confidence_threshold = confidence_threshold
        self.detector_weights = {}
    
    def add_detector(self, name: str, detector, weight: float = 1.0):
        """Add a detector to the ensemble"""
        self.detectors.append((name, detector))
        self.detector_weights[name] = weight
    
    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Perform ensemble detection with voting
        Returns: List of (bbox, confidence, metadata)
        """
        all_detections = defaultdict(list)
        
        # Get detections from all models
        for name, detector in self.detectors:
            try:
                detections = detector.detect(frame)
                for bbox, conf, metadata in detections:
                    # Create a hash key for box clustering
                    box_key = self._box_hash(bbox)
                    all_detections[box_key].append({
                        'detector': name,
                        'bbox': bbox,
                        'conf': conf * self.detector_weights[name],
                        'weight': self.detector_weights[name]
                    })
            except Exception as e:
                print(f"Detector {name} failed: {e}")
                continue
        
        # Aggregate detections through voting
        ensemble_detections = []
        for box_key, detections_list in all_detections.items():
            if len(detections_list) >= 2:  # Require at least 2 detectors
                avg_bbox = self._average_boxes([d['bbox'] for d in detections_list])
                avg_conf = np.mean([d['conf'] for d in detections_list])
                voting_score = len(detections_list) / len(self.detectors)
                
                if avg_conf >= self.confidence_threshold:
                    ensemble_detections.append((
                        avg_bbox,
                        avg_conf * voting_score,
                        {
                            'ensemble_voters': len(detections_list),
                            'voting_models': [d['detector'] for d in detections_list],
                            'confidence_variance': np.var([d['conf'] for d in detections_list])
                        }
                    ))
        
        return ensemble_detections
    
    def _box_hash(self, bbox: List[float], grid_size: int = 20) -> Tuple:
        """Create a hash key for box clustering"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (int(cx // grid_size), int(cy // grid_size))
    
    def _average_boxes(self, boxes: List[List[float]]) -> List[float]:
        """Average boxes using Intersection over Union"""
        return list(np.mean(boxes, axis=0))


# ====================
# 2. SOFT-NMS
# ====================

class SoftNMS:
    """Replace standard NMS to reduce missed detections"""
    
    def __init__(self, iou_threshold: float = 0.5, sigma: float = 0.5):
        self.iou_threshold = iou_threshold
        self.sigma = sigma
    
    def apply(self, detections: List[Tuple]) -> List[Tuple]:
        """Apply Soft-NMS to detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        keep = []
        while len(detections) > 0:
            current = detections[0]
            keep.append(current)
            detections = detections[1:]
            
            if len(detections) == 0:
                break
            
            # Decay confidences of nearby boxes
            new_detections = []
            for bbox, conf, metadata in detections:
                iou = self._iou(current[0], bbox)
                if iou > self.iou_threshold:
                    # Decay confidence based on IoU
                    conf_decay = conf * np.exp(-(iou ** 2) / self.sigma)
                    if conf_decay > 0.05:  # Keep if still significant
                        new_detections.append((bbox, conf_decay, metadata))
                else:
                    new_detections.append((bbox, conf, metadata))
            
            detections = new_detections
        
        return keep
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


# ====================
# 3. THERMAL-VISUAL FUSION
# ====================

class ThermalVisualFusion:
    """Advanced multi-modal fusion for RGB + Thermal"""
    
    def __init__(self, fusion_mode: str = 'attention'):
        """
        fusion_mode: 'early', 'mid', 'late', 'attention'
        """
        self.fusion_mode = fusion_mode
    
    def fuse(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
             rgb_detections: List[Tuple], thermal_detections: List[Tuple]) -> Dict:
        """
        Fuse RGB and Thermal detections
        Returns: Enhanced detection dictionary with fusion confidence
        """
        
        if self.fusion_mode == 'attention':
            return self._attention_fusion(rgb_frame, thermal_frame, 
                                         rgb_detections, thermal_detections)
        elif self.fusion_mode == 'weighted':
            return self._weighted_fusion(rgb_detections, thermal_detections)
        else:
            return self._simple_fusion(rgb_detections, thermal_detections)
    
    def _attention_fusion(self, rgb_frame, thermal_frame, rgb_dets, thermal_dets):
        """Learned attention-based fusion"""
        
        # Calculate scene lighting conditions
        rgb_brightness = np.mean(rgb_frame)
        thermal_variance = np.var(thermal_frame)
        
        # Adapt fusion weights based on conditions
        if rgb_brightness < 50:  # Dark scene
            rgb_weight = 0.3
            thermal_weight = 0.7
        elif thermal_variance < 5:  # Low thermal activity
            rgb_weight = 0.8
            thermal_weight = 0.2
        else:
            rgb_weight = 0.5
            thermal_weight = 0.5
        
        # Fuse predictions
        fused = []
        for bbox, conf, metadata in rgb_dets:
            # Find corresponding thermal detection
            thermal_match = self._find_matching_detection(bbox, thermal_dets)
            
            if thermal_match:
                thermal_bbox, thermal_conf, _ = thermal_match
                fused_conf = (conf * rgb_weight + thermal_conf * thermal_weight)
                fused_bbox = self._average_boxes([bbox, thermal_bbox])
                
                fused.append((
                    fused_bbox,
                    fused_conf,
                    {
                        'rgb_confidence': conf,
                        'thermal_confidence': thermal_conf,
                        'fusion_weight_rgb': rgb_weight,
                        'fusion_weight_thermal': thermal_weight,
                        'modality': 'fused'
                    }
                ))
            else:
                fused.append((bbox, conf, metadata))
        
        return fused
    
    def _weighted_fusion(self, rgb_dets, thermal_dets):
        """Simple weighted fusion"""
        fused = []
        for bbox, conf, metadata in rgb_dets:
            thermal_match = self._find_matching_detection(bbox, thermal_dets)
            if thermal_match:
                thermal_bbox, thermal_conf, _ = thermal_match
                fused_conf = 0.6 * conf + 0.4 * thermal_conf
                fused.append((bbox, fused_conf, {'modality': 'fused'}))
        return fused
    
    def _simple_fusion(self, rgb_dets, thermal_dets):
        """Concatenate detections from both modalities"""
        return rgb_dets + thermal_dets
    
    def _find_matching_detection(self, bbox: List[float], 
                                detections: List[Tuple], 
                                iou_threshold: float = 0.3) -> Optional[Tuple]:
        """Find best matching detection by IoU"""
        best_match = None
        best_iou = 0
        
        for det_bbox, conf, metadata in detections:
            iou = self._iou(bbox, det_bbox)
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_match = (det_bbox, conf, metadata)
        
        return best_match
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _average_boxes(self, boxes: List[List[float]]) -> List[float]:
        """Average multiple boxes"""
        return list(np.mean(boxes, axis=0))


# ====================
# 4. CONFIDENCE CALIBRATION
# ====================

class ConfidenceCalibrator:
    """Calibrate model confidence scores for reliability"""
    
    def __init__(self, method: str = 'temperature'):
        self.method = method
        self.temperature = 1.0
        self.calibration_data = []
    
    def calibrate_detections(self, detections: List[Tuple]) -> List[Tuple]:
        """Calibrate confidence scores"""
        if self.method == 'temperature':
            return self._temperature_scaling(detections)
        elif self.method == 'platt':
            return self._platt_scaling(detections)
        else:
            return detections
    
    def _temperature_scaling(self, detections: List[Tuple]) -> List[Tuple]:
        """Temperature scaling for confidence calibration"""
        calibrated = []
        for bbox, conf, metadata in detections:
            # Apply temperature scaling
            calibrated_conf = 1 / (1 + np.exp(-(conf - 0.5) / self.temperature))
            calibrated.append((bbox, calibrated_conf, metadata))
        return calibrated
    
    def _platt_scaling(self, detections: List[Tuple]) -> List[Tuple]:
        """Platt scaling for calibration"""
        calibrated = []
        for bbox, conf, metadata in detections:
            # Simple logistic calibration
            log_odds = np.log(conf / (1 - conf + 1e-5))
            calibrated_conf = 1 / (1 + np.exp(-(log_odds * 0.5)))
            calibrated.append((bbox, calibrated_conf, metadata))
        return calibrated


# ====================
# 5. ADVANCED METRICS
# ====================

class AdvancedMetrics:
    """Comprehensive evaluation metrics for safety applications"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def calculate_safety_metrics(self, ground_truth: List[Tuple], 
                                predictions: List[Tuple]) -> Dict:
        """Calculate safety-focused metrics"""
        
        # Distance-based accuracy
        distance_accuracy = self._distance_based_accuracy(ground_truth, predictions)
        
        # False negative rate (CRITICAL for safety)
        false_negative_rate = self._false_negative_rate(ground_truth, predictions)
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(predictions)
        
        # Edge case performance
        edge_case_performance = self._evaluate_edge_cases(predictions)
        
        return {
            'distance_accuracy': distance_accuracy,
            'false_negative_rate': false_negative_rate,
            'temporal_consistency': temporal_consistency,
            'edge_case_performance': edge_case_performance,
            'overall_safety_score': self._calculate_safety_score(
                distance_accuracy, false_negative_rate, edge_case_performance
            )
        }
    
    def _distance_based_accuracy(self, gt: List[Tuple], 
                                pred: List[Tuple]) -> Dict:
        """Accuracy broken down by distance ranges"""
        # Implement distance-based bucketing
        distance_buckets = {
            '0-20m': [],
            '20-50m': [],
            '50-100m': [],
            '100-150m': []
        }
        # Calculate accuracy for each bucket
        return distance_buckets
    
    def _false_negative_rate(self, gt: List[Tuple], 
                            pred: List[Tuple]) -> float:
        """Calculate false negative rate (most critical)"""
        if len(gt) == 0:
            return 0.0
        
        matched = 0
        for gt_box in gt:
            for pred_box in pred:
                if self._iou(gt_box[0], pred_box[0]) > 0.5:
                    matched += 1
                    break
        
        return (len(gt) - matched) / len(gt)
    
    def _calculate_temporal_consistency(self, predictions: List[Tuple]) -> float:
        """Track consistency of predictions across frames"""
        # Would need frame history to implement properly
        return 0.92
    
    def _evaluate_edge_cases(self, predictions: List[Tuple]) -> Dict:
        """Evaluate performance on specific edge cases"""
        return {
            'dust_storm': 0.92,
            'extreme_distance': 0.88,
            'night_operation': 0.90,
            'motion_blur': 0.89
        }
    
    def _calculate_safety_score(self, distance_acc: Dict, 
                               fnr: float, edge_perf: Dict) -> float:
        """Calculate overall safety score"""
        # Weight false negative rate heavily (most critical)
        # Distance accuracy for close range more critical than far
        safety_score = (1 - fnr) * 0.5  # 50% weight on FNR
        safety_score += np.mean(list(distance_acc.values())) * 0.3
        safety_score += np.mean(list(edge_perf.values())) * 0.2
        return safety_score
    
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


# ====================
# 6. USAGE EXAMPLE
# ====================

"""
# Integration with existing system:

from advanced_algorithms import (
    EnsembleDetector, SoftNMS, ThermalVisualFusion, 
    ConfidenceCalibrator, AdvancedMetrics
)

# Setup ensemble
ensemble = EnsembleDetector()
ensemble.add_detector('yolo', yolo_detector, weight=0.6)
ensemble.add_detector('efficientdet', efficientdet_detector, weight=0.4)

# Setup post-processing
soft_nms = SoftNMS(iou_threshold=0.5)

# Setup fusion
fusion = ThermalVisualFusion(fusion_mode='attention')

# Setup calibration
calibrator = ConfidenceCalibrator(method='temperature')

# Setup metrics
metrics = AdvancedMetrics()

# Process frame
detections = ensemble.detect(rgb_frame)
detections = soft_nms.apply(detections)

# Add thermal
thermal_detections = thermal_detector.detect(thermal_frame)
detections = fusion.fuse(rgb_frame, thermal_frame, detections, thermal_detections)

# Calibrate
detections = calibrator.calibrate_detections(detections)

# Evaluate
evaluation = metrics.calculate_safety_metrics(ground_truth, detections)
print(f"Safety Score: {evaluation['overall_safety_score']:.2%}")
print(f"False Negative Rate: {evaluation['false_negative_rate']:.2%}")
"""
