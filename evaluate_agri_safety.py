import os
import json
import logging
import numpy as np
from pycocotools.coco import COCO
import cv2
from detection import ObjectDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def draw_missed_boxes(image, missed_boxes):
    """
    Annotate missed ground truth boxes in red for false negative analysis.
    """
    annotated = image.copy()
    for bbox in missed_boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            annotated,
            'MISSED',
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return annotated


def evaluate_agri_safety(annotations_path, images_dir, output_dir='outputs/false_negatives', iou_threshold=0.5):
    """
    Evaluate detection on agricultural safety dataset.
    
    Args:
        annotations_path (str): Path to COCO annotations JSON.
        images_dir (str): Directory containing images.
        output_dir (str): Directory to save missed images.
        iou_threshold (float): IoU threshold for matching.
    """
    # Load COCO
    coco = COCO(annotations_path)
    person_cat_id = coco.getCatIds(catNms=['person'])[0]
    img_ids = coco.getImgIds(catIds=[person_cat_id])
    
    logging.info(f"Loaded {len(img_ids)} images with persons")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = ObjectDetector()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    missed_log = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            logging.warning(f"Image not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            logging.warning(f"Failed to load image: {img_path}")
            continue
        
        # Get GT bboxes for persons
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x,y,w,h] -> convert to [x1,y1,x2,y2]
        gt_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in gt_boxes]
        
        # Run detection
        detections = detector.detect(image)  # list of (bbox, conf)
        pred_boxes = [d[0] for d in detections]  # already [x1,y1,x2,y2]
        
        # Match predictions to GT
        matched_gt = set()
        matched_pred = set()
        
        for i, gt in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            for j, pred in enumerate(pred_boxes):
                if j in matched_pred:
                    continue
                iou = compute_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
            if best_iou >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
        
        tp = len(matched_gt)
        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Log missed cases
        if fn > 0:
            missed_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
            logging.info(f"Image {img_info['file_name']}: FN={fn}, GT={len(gt_boxes)}, Pred={len(pred_boxes)}")
            missed_log.append({
                'image': img_info['file_name'],
                'fn': fn,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'missed_gt': missed_boxes
            })

            annotated = draw_missed_boxes(image, missed_boxes)
            out_path = os.path.join(output_dir, img_info['file_name'])
            cv2.imwrite(out_path, annotated)
            logging.info(f"Saved false negative visualization: {out_path}")
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fnr = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    logging.info("Evaluation Results:")
    logging.info(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"False Negative Rate: {fnr:.4f}")
    
    # Save missed log
    with open(os.path.join(output_dir, 'missed_log.json'), 'w') as f:
        json.dump(missed_log, f, indent=2)
    
    logging.info(f"Missed images saved to {output_dir}, log saved as missed_log.json")

if __name__ == "__main__":
    # Example usage
    annotations_path = 'data/annotations.json'  # Adjust path
    images_dir = 'data/images'  # Adjust path
    evaluate_agri_safety(annotations_path, images_dir)