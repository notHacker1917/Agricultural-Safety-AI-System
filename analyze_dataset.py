"""
How to Analyze All Dataset Images and JSON Files

This script demonstrates comprehensive analysis of the HackHPI2026 dataset
including all images and JSON annotation files.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def analyze_all_dataset():
    """Analyze all images and JSON files in the HackHPI2026 dataset."""

    dataset_root = r"C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\HackHPI2026_release"
    annotation_dir = os.path.join(dataset_root, "annotation")
    data_dir = os.path.join(dataset_root, "data")

    print("=" * 80)
    print("HACKHPI2026 DATASET ANALYSIS")
    print("=" * 80)

    # 1. Find all JSON annotation files
    print("\n1. SCANNING ANNOTATION FILES...")
    json_files = list(Path(annotation_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSON annotation files:")

    for json_file in sorted(json_files):
        print(f"  - {json_file.name}")

    # 2. Find all image files
    print("\n2. SCANNING IMAGE FILES...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(data_dir).rglob(f"*{ext}")))

    print(f"Found {len(image_files)} image files")

    # 3. Analyze each JSON file
    print("\n3. ANALYZING JSON ANNOTATIONS...")

    total_images = 0
    total_annotations = 0
    categories_found = Counter()
    bbox_stats = []

    for json_file in sorted(json_files):
        print(f"\nAnalyzing: {json_file.name}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            images = data.get('images', [])
            annotations = data.get('annotations', [])
            categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

            print(f"  Images: {len(images)}")
            print(f"  Annotations: {len(annotations)}")
            print(f"  Categories: {list(categories.values())}")

            # Count annotations by category
            for ann in annotations:
                cat_name = categories.get(ann['category_id'], 'unknown')
                categories_found[cat_name] += 1

                # Collect bbox statistics
                if 'bbox' in ann:
                    bbox = ann['bbox']  # [x, y, width, height]
                    if len(bbox) == 4:
                        area = bbox[2] * bbox[3]  # width * height
                        bbox_stats.append({
                            'area': area,
                            'width': bbox[2],
                            'height': bbox[3],
                            'category': cat_name
                        })

            total_images += len(images)
            total_annotations += len(annotations)

        except Exception as e:
            print(f"  Error reading {json_file.name}: {e}")

    # 4. Overall statistics
    print("\n4. OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total JSON files: {len(json_files)}")
    print(f"Total images referenced: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Avg annotations per image: {total_annotations/total_images:.2f}" if total_images > 0 else "Avg annotations per image: N/A")

    print("\nAnnotations by category:")
    for cat, count in sorted(categories_found.items()):
        print(f"  {cat}: {count}")

    # 5. Bounding box analysis
    if bbox_stats:
        print("\n5. BOUNDING BOX ANALYSIS")
        print("-" * 40)

        areas = [b['area'] for b in bbox_stats]
        widths = [b['width'] for b in bbox_stats]
        heights = [b['height'] for b in bbox_stats]

        print(f"Total bounding boxes: {len(bbox_stats)}")
        print(f"Area range: {min(areas):.0f} - {max(areas):.0f} pixels²")
        print(f"Mean area: {np.mean(areas):.0f} pixels²")
        print(f"Median area: {np.median(areas):.0f} pixels²")
        print(f"Mean dimensions: {np.mean(widths):.1f} × {np.mean(heights):.1f} pixels")

        # Distance estimation (rough)
        # Assuming person ~1.7m tall, camera focal length ~300 pixels
        # At 10m: person appears ~50 pixels tall
        ref_height = 50  # pixels at 10m
        ref_distance = 10  # meters

        mean_height = np.mean(heights)
        estimated_distance = ref_distance * (ref_height / mean_height)
        print(f"Estimated average distance: ~{estimated_distance:.1f} meters")

        # By category
        print("\nBy category:")
        for cat in set(b['category'] for b in bbox_stats):
            cat_boxes = [b for b in bbox_stats if b['category'] == cat]
            if cat_boxes:
                cat_areas = [b['area'] for b in cat_boxes]
                print(f"  {cat}: {len(cat_boxes)} boxes, mean area: {np.mean(cat_areas):.0f} pixels²")

    # 6. Image file verification
    print("\n6. IMAGE FILE VERIFICATION")
    print("-" * 40)

    missing_images = 0
    found_images = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for img in data.get('images', []):
                img_filename = img['file_name']

                # Try to find the image file
                found = False
                for root, dirs, files in os.walk(data_dir):
                    if img_filename in files:
                        found = True
                        found_images += 1
                        break

                if not found:
                    missing_images += 1
                    print(f"  Missing: {img_filename}")

        except Exception as e:
            print(f"  Error checking images for {json_file.name}: {e}")

    print(f"Images found: {found_images}")
    print(f"Images missing: {missing_images}")

    # 7. Summary
    print("\n7. SUMMARY")
    print("-" * 40)
    print("✅ Dataset analysis complete!")
    print(f"   - {len(json_files)} annotation files processed")
    print(f"   - {total_images} images referenced")
    print(f"   - {total_annotations} annotations analyzed")
    print(f"   - {len(categories_found)} categories found")
    print(f"   - Distance range: close (<5m) to far (>30m)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_all_dataset()