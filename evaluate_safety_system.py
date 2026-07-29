import os
import json
import shutil
import logging
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
import cv2
from detection import ObjectDetector
from segmentation_tracking import DeepSORTTracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def sort_image_ids(coco, img_ids):
    """Sort images by frame order if available, otherwise by filename."""
    img_infos = [coco.loadImgs(i)[0] for i in img_ids]
    if img_infos and 'frame_id' in img_infos[0]:
        sorted_infos = sorted(img_infos, key=lambda x: x['frame_id'])
    else:
        sorted_infos = sorted(img_infos, key=lambda x: x['file_name'])
    return [info['id'] for info in sorted_infos], {info['id']: idx for idx, info in enumerate(sorted_infos)}


def get_track_id_from_ann(ann, fallback_idx):
    """Return a persistent track id from annotation if available."""
    return ann.get('track_id') or ann.get('instance_id') or ann.get('person_id') or f"frame_{ann['image_id']}_obj_{fallback_idx}"


class SafetyEvaluator:
    """Evaluates safety detection and tracking reliability."""

    def __init__(self, model_path='yolov8n.pt', conf=0.1, iou_threshold=0.5):
        self.detector = ObjectDetector(model_path=model_path, conf=conf)
        self.tracker = DeepSORTTracker()
        self.iou_threshold = iou_threshold

    def evaluate(self, annotations_path, images_dir, output_dir='data/safety_eval', save_missed=True):
        coco = COCO(annotations_path)
        person_cat_id = coco.getCatIds(catNms=['person'])[0]
        img_ids = coco.getImgIds(catIds=[person_cat_id])
        sorted_img_ids, _ = sort_image_ids(coco, img_ids)

        logging.info(f"Loaded {len(sorted_img_ids)} person frames for safety evaluation")
        os.makedirs(output_dir, exist_ok=True)

        total_tp = total_fp = total_fn = 0
        gt_track_meta = {}
        gt_first_appearance = {}
        detection_latency = []
        id_switches = 0
        missed_log = []
        summary_images = []
        real_track_ids = False

        for frame_index, img_id in enumerate(sorted_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                logging.warning(f"Missing image: {img_path}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Unable to load image: {img_path}")
                continue

            gt_ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
            gt_anns = coco.loadAnns(gt_ann_ids)
            gt_objects = []
            for idx, ann in enumerate(gt_anns):
                bbox = ann['bbox']
                track_id = get_track_id_from_ann(ann, idx)
                if 'track_id' in ann or 'instance_id' in ann or 'person_id' in ann:
                    real_track_ids = True
                if track_id not in gt_first_appearance:
                    gt_first_appearance[track_id] = frame_index
                gt_objects.append({
                    'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    'track_id': track_id,
                })

            if not gt_objects:
                continue

            detections = self.detector.detect(image)
            tracked = self.tracker.update(detections, frame=image)
            pred_entries = []
            for track_id, (bbox, conf, is_predicted, is_occluded, occlusion_duration) in tracked.items():
                pred_entries.append({
                    'bbox': bbox,
                    'track_id': track_id,
                    'conf': conf,
                    'is_predicted': is_predicted,
                    'is_occluded': is_occluded,
                    'occlusion_duration': occlusion_duration,
                })

            matches = self._match_objects(gt_objects, pred_entries)
            matched_gt = set()
            matched_pred = set()

            for match in matches:
                if match['iou'] >= self.iou_threshold:
                    matched_gt.add(match['gt_idx'])
                    matched_pred.add(match['pred_idx'])
                    total_tp += 1

                    gt_id = gt_objects[match['gt_idx']]['track_id']
                    pred_track_id = pred_entries[match['pred_idx']]['track_id']

                    if gt_id not in gt_track_meta:
                        first_seen = gt_first_appearance.get(gt_id, frame_index)
                        latency = frame_index - first_seen
                        gt_track_meta[gt_id] = {
                            'first_frame': first_seen,
                            'first_detected': frame_index,
                            'last_pred_track': pred_track_id,
                            'switches': 0,
                        }
                        detection_latency.append(latency)
                    else:
                        if gt_track_meta[gt_id]['last_pred_track'] is not None and gt_track_meta[gt_id]['last_pred_track'] != pred_track_id:
                            gt_track_meta[gt_id]['switches'] += 1
                            id_switches += 1
                        gt_track_meta[gt_id]['last_pred_track'] = pred_track_id
                        if gt_track_meta[gt_id]['first_detected'] is None:
                            latency = frame_index - gt_track_meta[gt_id]['first_frame']
                            detection_latency.append(latency)
                            gt_track_meta[gt_id]['first_detected'] = frame_index

            total_fp += len(pred_entries) - len(matched_pred)
            total_fn += len(gt_objects) - len(matched_gt)

            if len(gt_objects) - len(matched_gt) > 0 and save_missed:
                missed_gt = [gt_objects[i]['bbox'] for i in range(len(gt_objects)) if i not in matched_gt]
                missed_log.append({
                    'image': img_info['file_name'],
                    'fn': len(missed_gt),
                    'gt_count': len(gt_objects),
                    'pred_count': len(pred_entries),
                    'missed_gt': missed_gt,
                })
                shutil.copy(img_path, os.path.join(output_dir, img_info['file_name']))
                summary_images.append(img_info['file_name'])

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        fnr = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        avg_latency = float(np.mean(detection_latency)) if detection_latency else None
        median_latency = float(np.median(detection_latency)) if detection_latency else None
        total_tracks = len(gt_track_meta)
        avg_switches = float(id_switches) / total_tracks if total_tracks > 0 else None

        summary = {
            'total_images': len(sorted_img_ids),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': precision,
            'recall': recall,
            'false_negative_rate': fnr,
            'time_to_detect_avg_frames': avg_latency,
            'time_to_detect_median_frames': median_latency,
            'total_tracks': total_tracks,
            'total_id_switches': id_switches,
            'avg_id_switches_per_track': avg_switches,
            'use_real_track_ids': real_track_ids,
            'missed_images': len(missed_log),
            'top_failure_cases': summary_images[:10],
            'insights': self._generate_insights(precision, recall, fnr, avg_latency, id_switches, real_track_ids),
        }

        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(self._format_summary(summary))

        with open(os.path.join(output_dir, 'missed_log.json'), 'w') as f:
            json.dump(missed_log, f, indent=2)

        logging.info('Safety evaluation completed')
        logging.info('\n' + self._format_summary(summary))
        return summary

    def _match_objects(self, gt_objects, pred_entries):
        matches = []
        used_pred = set()
        for gt_idx, gt in enumerate(gt_objects):
            best_iou = 0
            best_pred = None
            for pred_idx, pred in enumerate(pred_entries):
                if pred_idx in used_pred:
                    continue
                iou = compute_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = {
                        'gt_idx': gt_idx,
                        'pred_idx': pred_idx,
                        'pred_track_id': pred['track_id'],
                        'iou': iou,
                    }
            if best_pred is not None:
                matches.append(best_pred)
                used_pred.add(best_pred['pred_idx'])
        return matches

    def _generate_insights(self, precision, recall, fnr, avg_latency, id_switches, real_track_ids):
        insights = []
        if fnr > 0.2:
            insights.append('High false negative rate indicates missed humans in safety-critical scenes.')
        else:
            insights.append('False negative rate is acceptable for the evaluated data.')
        if precision < 0.8:
            insights.append('Precision is low; this may indicate false alarms from poles, scarecrows, or crop clutter.')
        if avg_latency is None:
            insights.append('Time-to-detect metrics are unavailable; ensure track IDs are present in annotation metadata or sequence annotations.')
        elif avg_latency > 2.0:
            insights.append('Detection latency is high; this suggests delayed response before the person is first detected.')
        else:
            insights.append('Time-to-detect is low, supporting timely warnings in live deployment.')
        if not real_track_ids:
            insights.append('Track IDs were not available in annotations; tracking stability metrics may be approximate.')
        if id_switches > 5:
            insights.append('Many ID switches were observed; tracking stability should be improved with stronger appearance or motion cues.')
        else:
            insights.append('Tracking stability appears reasonable for the evaluated sequence.')
        insights.append('For life-critical reliability, focus on occlusions, low-light frames, and reducing missed person detections.')
        return insights

    def _format_summary(self, summary):
        lines = [
            'Safety Evaluation Summary',
            '------------------------',
            f"Total images evaluated: {summary['total_images']}",
            f"True positives: {summary['total_tp']}",
            f"False positives: {summary['total_fp']}",
            f"False negatives: {summary['total_fn']}",
            f"Precision: {summary['precision']:.4f}",
            f"Recall: {summary['recall']:.4f}",
            f"False Negative Rate: {summary['false_negative_rate']:.4f}",
            f"Avg time-to-detect (frames): {summary['time_to_detect_avg_frames']}",
            f"Median time-to-detect (frames): {summary['time_to_detect_median_frames']}",
            f"Total tracks analyzed: {summary['total_tracks']}",
            f"Total ID switches: {summary['total_id_switches']}",
            f"Avg ID switches per track: {summary['avg_id_switches_per_track']}",
            f"Missed images saved: {summary['missed_images']}",
            'Insights:',
        ]
        for insight in summary['insights']:
            lines.append(f"- {insight}")
        lines.append('Top failure cases:')
        for name in summary['top_failure_cases']:
            lines.append(f"- {name}")
        return '\n'.join(lines)


if __name__ == '__main__':
    annotations_path = 'data/annotations.json'  # Adjust path
    images_dir = 'data/images'  # Adjust path
    evaluator = SafetyEvaluator()
    evaluator.evaluate(annotations_path, images_dir)
