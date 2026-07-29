#!/usr/bin/env python3
"""
Convert HackHPI2026_release dataset from COCO to YOLO format
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

def convert_coco_to_yolo():
    # Create processed directory structure in data folder
    processed_dir = Path('data/processed')
    images_dir = processed_dir / 'images'
    labels_dir = processed_dir / 'labels'

    # Create directories using os.makedirs with absolute paths
    import os
    print(f"Creating directories: {images_dir}, {labels_dir}")
    
    # Use absolute path
    abs_processed = Path.cwd() / processed_dir
    abs_images = Path.cwd() / images_dir
    abs_labels = Path.cwd() / labels_dir
    
    print(f"Absolute paths: {abs_processed}, {abs_images}, {abs_labels}")
    
    # Create parent directory first
    os.makedirs(str(abs_processed), exist_ok=True)
    print(f"Created {abs_processed}")
    
    # Then create subdirectories
    os.makedirs(str(abs_images), exist_ok=True)
    print(f"Created {abs_images}")
    
    os.makedirs(str(abs_labels), exist_ok=True)
    print(f"Created {abs_labels}")

    print('All processed directories created')

    # Convert COCO to YOLO
    annotations_dir = Path('C:/Users/hs735.COLTSMOKE/OneDrive/Documents/Hackathon/HackHPI2026_release/annotation')
    images_base_dir = Path('C:/Users/hs735.COLTSMOKE/OneDrive/Documents/Hackathon/HackHPI2026_release/data')

    json_files = list(annotations_dir.glob('**/*.json'))
    print(f'Found {len(json_files)} annotation files')

    total_converted = 0
    for json_file in json_files:
        print(f'Processing {json_file.name}')

        with open(json_file, 'r') as f:
            coco_data = json.load(f)

        if 'images' not in coco_data or 'annotations' not in coco_data:
            print(f'Skipping {json_file.name} - not COCO format')
            continue

        images = {img['id']: img for img in coco_data['images']}
        annotations = defaultdict(list)

        # Group annotations by image
        for ann in coco_data['annotations']:
            if ann.get('category_id') in [0, 1]:  # person class
                annotations[ann['image_id']].append(ann)

        for img_id, img_info in images.items():
            img_anns = annotations.get(img_id, [])
            if not img_anns:
                continue

            # Create YOLO label file
            label_file = abs_labels / f"{Path(img_info['file_name']).stem}.txt"
            img_width, img_height = img_info['width'], img_info['height']

            yolo_lines = []
            for ann in img_anns:
                bbox = ann['bbox']  # [x, y, w, h]
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))

            # Copy image
            # Find the image in the dataset structure
            found = False
            for env_dir in images_base_dir.iterdir():
                if env_dir.is_dir():
                    for sub_dir in env_dir.iterdir():
                        if sub_dir.is_dir():
                            img_path = sub_dir / img_info['file_name']
                            if img_path.exists():
                                dst_img = abs_images / img_info['file_name']
                                shutil.copy2(img_path, dst_img)
                                found = True
                                break
                    if found:
                        break

            if found:
                total_converted += 1
            else:
                print(f'Image not found: {img_info["file_name"]}')

    print(f'Converted {total_converted} images with annotations')

    # Update data.yaml to point to processed data
    data_yaml_content = f"""# Agricultural Safety Dataset Configuration
# Generated for HackHPI2026_release dataset

path: data/processed
train: images
val: images

# Classes
nc: 1
names: ['person']
"""

    with open('data.yaml', 'w') as f:
        f.write(data_yaml_content)

    print('Updated data.yaml configuration')

if __name__ == '__main__':
    convert_coco_to_yolo()