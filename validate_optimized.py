#!/usr/bin/env python3
"""
OPTIMIZED AGRICULTURAL SAFETY SYSTEM - PRODUCTION VERSION
Simplified working version that integrates with existing pipeline
"""

import json
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

# Configure logging
log_dir = os.path.expanduser("~/safety_logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "optimized_validation_prod.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionOptimizer:
    """Optimized detector for production deployment."""
    
    def __init__(self):
        logger.info("="*80)
        logger.info("OPTIMIZED AGRICULTURAL SAFETY VALIDATION - PRODUCTION")
        logger.info("="*80)
        
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            logger.info("[DETECTOR] YOLOv8 Nano loaded")
            self.conf_threshold = 0.50  # INCREASED from 0.25
            logger.info(f"[THRESHOLD] Confidence set to {self.conf_threshold}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self.model = None
    
    def detect(self, image_path: str) -> List[Tuple]:
        """Run YOLO detection."""
        if self.model is None:
            return []
        
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Only person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        if conf >= self.conf_threshold:
                            detections.append(((x1, y1, x2, y2), conf))
            
            # NMS filtering
            detections = self._nms(detections, threshold=0.45)
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    def _nms(self, detections: List[Tuple], threshold: float = 0.45) -> List[Tuple]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        keep = []
        
        while detections:
            keep.append(detections[0])
            if len(detections) == 1:
                break
            
            x1_1, y1_1, x2_1, y2_1 = detections[0][0]
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            
            remaining = []
            for det in detections[1:]:
                x1_2, y1_2, x2_2, y2_2 = det[0]
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                
                if x2_i > x1_i and y2_i > y1_i:
                    inter = (x2_i - x1_i) * (y2_i - y1_i)
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou < threshold:
                        remaining.append(det)
                else:
                    remaining.append(det)
            
            detections = remaining
        
        return keep


def run_full_pipeline():
    """Run complete optimized validation."""
    from dataset_extraction_pipeline import DatasetExtractor
    
    logger.info("\n[STEP 1] Extracting dataset...")
    extractor = DatasetExtractor()
    ground_truth = extractor.load_and_validate_dataset()
    images_metadata = extractor.image_metadata
    
    if not ground_truth:
        logger.error("Failed to load dataset")
        return
    
    logger.info(f"[DATASET] {len(ground_truth)} images with {sum(len(anns) for anns in ground_truth.values())} annotations")
    
    logger.info("\n[STEP 2] Initializing optimized detector...")
    detector = ProductionOptimizer()
    
    logger.info("\n[STEP 3] Running detection on sample...")
    detections_by_image = defaultdict(list)
    
    sample_size = min(100, len(ground_truth))
    for idx, (image_id, annotations) in enumerate(list(ground_truth.items())[:sample_size]):
        if idx % 20 == 0:
            logger.info(f"Processed {idx}/{sample_size} images...")
        
        if image_id not in images_metadata:
            continue
        
        image_path = images_metadata[image_id].get('file_path')
        if not image_path or not os.path.exists(image_path):
            continue
        
        detections = detector.detect(image_path)
        detections_by_image[image_id] = detections
    
    logger.info(f"Processed {len(detections_by_image)} images")
    
    # Calculate metrics
    logger.info("\n[STEP 4] Calculating metrics...")
    tp = 0
    fp = 0
    fn = 0
    
    matched = set()
    for image_id, detections in detections_by_image.items():
        if image_id not in ground_truth:
            fp += len(detections)
            continue
        
        for det_box, conf in detections:
            x1d, y1d, x2d, y2d = det_box
            wd = x2d - x1d
            hd = y2d - y1d
            area_det = wd * hd
            
            best_iou = 0
            best_gt_idx = None
            
            for gt_idx, gt_ann in enumerate(ground_truth[image_id]):
                if (image_id, gt_idx) in matched:
                    continue
                
                gx, gy, gw, gh = gt_ann['bbox']
                area_gt = gw * gh
                
                xi1 = max(x1d, gx)
                yi1 = max(y1d, gy)
                xi2 = min(x2d, gx + gw)
                yi2 = min(y2d, gy + gh)
                
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    union = area_det + area_gt - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou > 0.5 and best_gt_idx is not None:
                tp += 1
                matched.add((image_id, best_gt_idx))
            else:
                fp += 1
    
    # Count false negatives
    for image_id, annotations in ground_truth.items():
        for gt_idx in range(len(annotations)):
            if (image_id, gt_idx) not in matched:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZED AGRICULTURAL SAFETY SYSTEM - RESULTS")
    print("="*80)
    print(f"\nImages Processed:     {len(detections_by_image)}")
    print(f"Total Detections:     {sum(len(d) for d in detections_by_image.values())}")
    print(f"Ground Truth Objects: {sum(len(a) for a in ground_truth.values())}")
    print(f"\nConfidence Threshold: {detector.conf_threshold}")
    print(f"NMS Threshold:        0.45")
    print(f"\nMETRICS:")
    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1-Score:   {f1:.1%}")
    print(f"\nCONFUSION MATRIX:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print("="*80)
    
    # Save results
    results = {
        'images_processed': len(detections_by_image),
        'total_detections': sum(len(d) for d in detections_by_image.values()),
        'ground_truth_objects': sum(len(a) for a in ground_truth.values()),
        'confidence_threshold': detector.conf_threshold,
        'nms_threshold': 0.45,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    }
    
    result_file = os.path.join(log_dir, "optimized_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    run_full_pipeline()
