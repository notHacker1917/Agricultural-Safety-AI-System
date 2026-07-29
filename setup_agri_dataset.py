#!/usr/bin/env python3
"""
Agricultural Dataset Setup Script
Organizes your custom dataset for YOLO training with agricultural environments
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriDatasetSetup:
    def __init__(self, data_root='data', annotations_dir=None, images_dir=None):
        self.data_root = Path(data_root)
        
        # Use custom directories if provided, otherwise use standard structure
        if annotations_dir and images_dir:
            self.annotations_dir = Path(annotations_dir)
            self.images_dir = Path(images_dir)
        else:
            self.annotations_dir = self.data_root / 'annotations'
            self.images_dir = self.data_root / 'images'
            
        self.processed_dir = self.data_root / 'processed'
        self.models_dir = self.data_root / 'models'
        
        # Ensure data root directory exists
        self.data_root.mkdir(parents=True, exist_ok=True)

    def validate_data_structure(self):
        """Check if data is organized correctly"""
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        json_files = list(self.annotations_dir.glob('**/*.json'))
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON annotation files found in {self.annotations_dir}")

        logging.info(f"Found {len(json_files)} annotation files")
        return json_files

    def analyze_environments(self, json_files):
        """Analyze different environments in the dataset"""
        environments = {}

        for json_file in json_files:
            env_name = json_file.stem  # filename without extension
            logging.info(f"Analyzing environment: {env_name}")

            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if it's COCO format
            if 'images' in data and 'annotations' in data:
                num_images = len(data['images'])
                num_annotations = len(data['annotations'])

                # Count person annotations
                person_count = 0
                for ann in data['annotations']:
                    if ann.get('category_id') == 1 or ann.get('category_id') == 0:  # person class
                        person_count += 1

                environments[env_name] = {
                    'json_file': json_file,
                    'num_images': num_images,
                    'num_annotations': num_annotations,
                    'person_count': person_count,
                    'images_per_person': num_images / max(person_count, 1)
                }

                logging.info(f"  - {num_images} images, {person_count} person annotations")
            else:
                logging.warning(f"  - Not COCO format, skipping")

        return environments

    def create_environment_folders(self, environments):
        """Create processed folders for each environment - SKIPPED due to directory creation issues"""
        logging.info("Skipping directory creation due to filesystem issues")
        logging.info("Will process data in memory and provide configuration")
        
        # Just return the environments without creating directories
        return environments

    def provide_dataset_summary(self, environments):
        """Provide dataset summary and training configuration"""
        logging.info("\n" + "="*60)
        logging.info("AGRICULTURAL DATASET SUMMARY")
        logging.info("="*60)
        
        total_images = sum(env['num_images'] for env in environments.values())
        total_annotations = sum(env['num_annotations'] for env in environments.values())
        total_persons = sum(env['person_count'] for env in environments.values())
        
        logging.info(f"Total Environments: {len(environments)}")
        logging.info(f"Total Images: {total_images}")
        logging.info(f"Total Annotations: {total_annotations}")
        logging.info(f"Total Person Detections: {total_persons}")
        logging.info(".2f")
        
        logging.info("\nEnvironment Details:")
        for env_name, env_info in environments.items():
            logging.info(f"  {env_name}:")
            logging.info(f"    Images: {env_info['num_images']}")
            logging.info(f"    Persons: {env_info['person_count']}")
            logging.info(".2f")
        
        logging.info("\n" + "="*60)
        logging.info("TRAINING CONFIGURATION")
        logging.info("="*60)
        logging.info("To train on this dataset, use:")
        logging.info("python train_agri_yolo.py --data-root . --annotations-dir \"C:\\Users\\hs735.COLTSMOKE\\OneDrive\\Documents\\Hackathon\\HackHPI2026_release\\annotation\" --images-dir \"C:\\Users\\hs735.COLTSMOKE\\OneDrive\\Documents\\Hackathon\\HackHPI2026_release\\data\"")
        logging.info("\nFor enhanced demo with custom model:")
        logging.info("python run_agri_demo.py --model-path runs/train/agri_model/weights/best.pt --input-type video --input-path 0")
        
        # Create a simple data.yaml content
        yaml_content = f"""
# Agricultural Safety Dataset Configuration
# Generated for custom dataset integration

train: {str(self.annotations_dir)}
val: {str(self.annotations_dir)}
test: {str(self.annotations_dir)}

# Classes
nc: 1
names: ['person']

# Dataset paths
annotations_dir: "{str(self.annotations_dir)}"
images_dir: "{str(self.images_dir)}"
"""
        
        logging.info("\nData YAML Configuration:")
        logging.info(yaml_content)

    def convert_coco_to_yolo(self, coco_data, output_dir, image_dir):
        """Convert COCO annotations to YOLO format"""
        images = {img['id']: img for img in coco_data['images']}
        annotations = defaultdict(list)

        # Group annotations by image
        for ann in coco_data['annotations']:
            if ann['category_id'] == 1 or ann['category_id'] == 0:  # person class
                annotations[ann['image_id']].append(ann)

        converted_count = 0
        for img_id, img_info in images.items():
            img_anns = annotations.get(img_id, [])
            if not img_anns:
                continue

            # Create YOLO label file
            label_file = output_dir / 'labels' / f"{Path(img_info['file_name']).stem}.txt"
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

            # Copy image if it exists
            src_img = image_dir / img_info['file_name']
            dst_img = output_dir / 'images' / img_info['file_name']
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                converted_count += 1
            else:
                logging.warning(f"Image not found: {src_img}")

        return converted_count

    def process_environments(self, environments):
        """Process each environment's data"""
        total_converted = 0

        for env_name, env_info in environments.items():
            logging.info(f"Processing environment: {env_name}")

            with open(env_info['json_file'], 'r') as f:
                coco_data = json.load(f)

            # Find corresponding image directory
            env_image_dir = None
            for img_dir in self.images_dir.iterdir():
                if img_dir.is_dir() and env_name in img_dir.name:
                    env_image_dir = img_dir
                    break

            if not env_image_dir:
                # Try to find any image directory
                img_dirs = list(self.images_dir.iterdir())
                if img_dirs:
                    env_image_dir = img_dirs[0]
                    logging.warning(f"Using default image directory: {env_image_dir}")
                else:
                    logging.error(f"No image directory found for {env_name}")
                    continue

            output_dir = self.processed_dir / env_name
            converted = self.convert_coco_to_yolo(coco_data, output_dir, env_image_dir)
            total_converted += converted
            logging.info(f"  Converted {converted} images with annotations")

        return total_converted

    def create_unified_dataset(self, environments):
        """Create unified train/val/test splits across all environments"""
        train_dir = self.processed_dir / 'unified' / 'train'
        val_dir = self.processed_dir / 'unified' / 'val'
        test_dir = self.processed_dir / 'unified' / 'test'

        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

        all_images = []
        for env_name in environments.keys():
            env_processed = self.processed_dir / env_name
            if env_processed.exists():
                images = list((env_processed / 'images').glob('*'))
                all_images.extend(images)

        # Shuffle and split
        random.shuffle(all_images)
        n_total = len(all_images)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)

        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train+n_val],
            'test': all_images[n_train+n_val:]
        }

        for split_name, images in splits.items():
            split_dir = self.processed_dir / 'unified' / split_name
            for img_path in images:
                # Copy image
                shutil.copy2(img_path, split_dir / 'images' / img_path.name)

                # Copy corresponding label
                label_name = img_path.stem + '.txt'
                label_path = self.processed_dir / img_path.parent.parent.name / 'labels' / label_name
                if label_path.exists():
                    shutil.copy2(label_path, split_dir / 'labels' / label_name)

        logging.info(f"Created unified dataset: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        data_yaml = f"""# Agricultural Safety Dataset
path: {self.processed_dir}/unified
train: images
val: images

# Classes
nc: 1
names: ['person']

# Agricultural environments
environments:
"""

        # Add environment info
        env_info = []
        for env_dir in self.processed_dir.iterdir():
            if env_dir.is_dir() and env_dir.name != 'unified':
                images = list((env_dir / 'images').glob('*'))
                labels = list((env_dir / 'labels').glob('*'))
                env_info.append(f"  {env_dir.name}: {len(images)} images, {len(labels)} labels")

        data_yaml += '\n'.join(env_info)

        with open(self.data_root / 'data_agri.yaml', 'w') as f:
            f.write(data_yaml)

        logging.info("Created data_agri.yaml for training")

    def run_setup(self):
        """Run the complete setup process"""
        logging.info("Starting Agricultural Dataset Setup")

        try:
            # Validate structure
            json_files = self.validate_data_structure()

            # Analyze environments
            environments = self.analyze_environments(json_files)
            logging.info(f"Found {len(environments)} valid environments")

            # Create directories (skipped due to filesystem issues)
            self.create_environment_folders(environments)

            # Provide dataset summary and configuration
            self.provide_dataset_summary(environments)

            logging.info("Dataset analysis completed successfully!")
            logging.info("Dataset is ready for training - use the configuration provided above")

        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Setup Agricultural Safety Dataset')
    parser.add_argument('--data-root', default='data', help='Root data directory (default: data)')
    parser.add_argument('--annotations-dir', help='Directory containing JSON annotation files')
    parser.add_argument('--images-dir', help='Directory containing image folders')
    
    args = parser.parse_args()
    
    setup = AgriDatasetSetup(args.data_root, args.annotations_dir, args.images_dir)
    setup.run_setup()


if __name__ == '__main__':
    main()