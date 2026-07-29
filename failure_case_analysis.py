import argparse
import json
import logging
import math
import os
from collections import defaultdict

import cv2
import numpy as np
from coco_loader import COCODataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FAILURE_CATEGORIES = [
    'occlusion_failure',
    'small_object_failure',
    'motion_blur_failure',
    'lighting_failure',
    'false_positive',
    'background_confusion',
]

SMALL_AREA_RATIO_THRESHOLD = 0.015
LOW_LIGHT_THRESHOLD = 70
MOTION_BLUR_THRESHOLD = 40
OCCLUSION_EDGE_DENSITY_THRESHOLD = 0.10
CRITICAL_BOTTOM_RATIO = 0.7


def compute_patch_metrics(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
    return mean_intensity, lap_var, edge_density


def crop_patch(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def classify_missed_detection(image, bbox):
    patch = crop_patch(image, bbox)
    if patch.size == 0:
        return 'background_confusion'

    mean_intensity, lap_var, edge_density = compute_patch_metrics(patch)
    area_ratio = (patch.shape[0] * patch.shape[1]) / (image.shape[0] * image.shape[1])

    if area_ratio < SMALL_AREA_RATIO_THRESHOLD:
        return 'small_object_failure'
    if mean_intensity < LOW_LIGHT_THRESHOLD:
        return 'lighting_failure'
    if lap_var < MOTION_BLUR_THRESHOLD:
        return 'motion_blur_failure'
    if edge_density > OCCLUSION_EDGE_DENSITY_THRESHOLD:
        return 'occlusion_failure'
    return 'background_confusion'


def is_critical_safety_failure(bbox, image_height):
    bottom_y = bbox[3]
    return (bottom_y / float(image_height)) >= CRITICAL_BOTTOM_RATIO


def annotate_failure_image(image, bbox, category, status_label, is_critical=False):
    annotated = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if status_label == 'FN' else (0, 255, 255)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    label = f'{status_label}: {category.replace("_", " ").upper()}'
    if is_critical:
        label += ' CRITICAL'
    cv2.putText(
        annotated,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def save_failure_image(image, image_name, category, output_dir, suffix):
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    filename = f'{base_name}_{suffix}_{category}.jpg'
    path = os.path.join(category_dir, filename)
    cv2.imwrite(path, image)
    return path


def compute_frame_metrics(frame_records):
    tp = sum(1 for r in frame_records if r['status'] == 'TP')
    fp = sum(1 for r in frame_records if r['status'] == 'FP')
    fn = sum(1 for r in frame_records if r['status'] == 'FN')
    return tp, fp, fn


def build_summary_table(category_stats, baseline_precision):
    rows = []
    for category, stats in category_stats.items():
        precision = stats['precision']
        drop = baseline_precision - precision if baseline_precision is not None else None
        rows.append({
            'category': category,
            'count': stats['count'],
            'false_negative_rate': stats.get('fn_rate', 0.0),
            'precision': precision,
            'precision_drop': drop,
        })
    return rows


def print_summary_table(rows):
    header = f"{'Category':<22}{'Count':>8}{'FNR':>10}{'Precision':>12}{'Drop':>10}"
    print(header)
    print('-' * len(header))
    for row in rows:
        drop_str = f"{row['precision_drop']:.3f}" if row['precision_drop'] is not None else 'N/A'
        print(
            f"{row['category']:<22}{row['count']:>8}{row['false_negative_rate']:>10.3f}"
            f"{row['precision']:>12.3f}{drop_str:>10}"
        )


def analyze_pipeline_failures(
    results_path,
    annotations_path,
    images_dir,
    output_dir='outputs/failure_cases',
    summary_path=None,
    high_risk_threshold=CRITICAL_BOTTOM_RATIO,
):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f'Results file not found: {results_path}')

    dataset = COCODataset(annotations_path, images_dir)
    with open(results_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frame_groups = defaultdict(list)
    for record in records:
        frame_groups[record['frame_id']].append(record)

    category_counts = {cat: 0 for cat in FAILURE_CATEGORIES}
    category_fn = {cat: 0 for cat in FAILURE_CATEGORIES}
    category_tp = {cat: 0 for cat in FAILURE_CATEGORIES}
    category_fp = {cat: 0 for cat in FAILURE_CATEGORIES}
    category_frame_ids = defaultdict(set)
    critical_cases = []
    annotated_paths = []

    baseline_tp = sum(1 for r in records if r['status'] == 'TP')
    baseline_fp = sum(1 for r in records if r['status'] == 'FP')
    baseline_fn = sum(1 for r in records if r['status'] == 'FN')
    baseline_precision = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0.0

    for frame_id, frame_records in frame_groups.items():
        image_id = frame_records[0].get('image_id')
        try:
            image, _ = dataset.get_image(image_id)
        except Exception:
            image = None

        image_name = dataset.images.get(image_id, {}).get('file_name') if image is not None else f'image_{image_id}'

        if image is None:
            logging.warning(f'Image not found for analysis: image_id={image_id}')
            continue

        frame_tp, frame_fp, frame_fn = compute_frame_metrics(frame_records)
        frame_precision = frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0.0

        for rec_idx, record in enumerate(frame_records):
            status = record['status']
            bbox = record['bbox']
            category = None
            critical = False
            if status == 'FN':
                category = classify_missed_detection(image, bbox)
                critical = is_critical_safety_failure(bbox, image.shape[0])
                category_fn[category] += 1
                category_counts[category] += 1
                category_frame_ids[category].add(frame_id)
                annotated = annotate_failure_image(image, bbox, category, status, critical)
                suffix = f'frame{frame_id:05d}_fn{rec_idx}'
                path = save_failure_image(annotated, image_name, category, output_dir, suffix)
                annotated_paths.append(path)
                if critical:
                    critical_cases.append({
                        'frame_id': frame_id,
                        'image_id': image_id,
                        'image_name': image_name,
                        'category': category,
                        'bbox': bbox,
                        'reason': 'high risk missed detection',
                    })
            elif status == 'FP':
                category = 'false_positive'
                category_fp[category] += 1
                category_counts[category] += 1
                category_frame_ids[category].add(frame_id)
                annotated = annotate_failure_image(image, bbox, category, status)
                suffix = f'frame{frame_id:05d}_fp{rec_idx}'
                path = save_failure_image(annotated, image_name, category, output_dir, suffix)
                annotated_paths.append(path)
            elif status == 'TP':
                continue
            else:
                continue

            # Track precision contributions by category frames
            if status == 'TP':
                category_tp[category] += 1
            if status == 'FP':
                category_fp[category] += 0

    category_stats = {}
    for category in FAILURE_CATEGORIES:
        count = category_counts[category]
        fn_count = category_fn.get(category, 0)
        relevant_frames = category_frame_ids[category]
        frame_precision_values = []
        for fid in relevant_frames:
            tp, fp, _ = compute_frame_metrics(frame_groups[fid])
            frame_precision_values.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        category_precision = sum(frame_precision_values) / len(frame_precision_values) if frame_precision_values else baseline_precision
        fn_rate = fn_count / baseline_fn if baseline_fn > 0 else 0.0
        category_stats[category] = {
            'count': count,
            'false_negative_rate': fn_rate,
            'precision': category_precision,
            'precision_drop': baseline_precision - category_precision,
            'frames_with_category': len(relevant_frames),
        }

    report = {
        'baseline': {
            'precision': baseline_precision,
            'recall': baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0.0,
            'false_negative_rate': baseline_fn / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0.0,
            'total_tp': baseline_tp,
            'total_fp': baseline_fp,
            'total_fn': baseline_fn,
        },
        'categories': category_stats,
        'critical_failures': critical_cases,
        'annotated_images': annotated_paths,
    }

    report_path = os.path.abspath(summary_path) if summary_path else os.path.join(output_dir, 'failure_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    logging.info(f'Failure analysis saved to {report_path}')
    print('\nFailure Analysis Summary')
    print('========================')
    print_summary_table([{
        'category': cat,
        'count': stats['count'],
        'false_negative_rate': stats['false_negative_rate'],
        'precision': stats['precision'],
        'precision_drop': stats['precision_drop'],
    } for cat, stats in category_stats.items()])

    print('\nCritical Safety Failures:')
    for case in critical_cases[:10]:
        print(f"frame={case['frame_id']} image={case['image_name']} category={case['category']} bbox={case['bbox']}")

    return report


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze failure cases from pipeline output')
    parser.add_argument('--results', default='outputs/inference/frame_results.json', help='Pipeline JSON results file')
    parser.add_argument('--annotations', default='data/annotations.json', help='COCO annotations JSON for image mapping')
    parser.add_argument('--images', default='data/images', help='Original image folder')
    parser.add_argument('--output', default='outputs/failure_cases', help='Output directory for failure cases')
    parser.add_argument('--summary', default='outputs/failure_report.json', help='Failure report JSON path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_pipeline_failures(
        results_path=args.results,
        annotations_path=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        summary_path=args.summary,
    )
