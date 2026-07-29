"""
DATASET EXTRACTION & PROCESSING

Extracts JSON annotations and images from HackHPI2026 dataset.
Prepares data for validation against our agricultural safety system.

Supports:
- JSON annotation parsing (COCO format)
- Image batch loading
- Ground truth evaluation
- Performance metrics calculation
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import cv2

log_dir = os.path.expanduser("~/safety_logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "dataset_extraction.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Single image with annotations."""
    image_id: int
    file_path: str
    width: int
    height: int
    annotations: List[Dict]  # Bounding boxes
    is_loaded: bool = False
    image_array: np.ndarray = None


@dataclass
class CocoDataset:
    """Complete COCO dataset."""
    images: Dict[int, ImageData]
    categories: Dict[int, str]
    info: Dict
    total_annotations: int


class DatasetExtractor:
    """Extract and process HackHPI2026 dataset."""
    
    DATASET_ROOT = r"C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\HackHPI2026_release"
    ANNOTATION_DIR = os.path.join(DATASET_ROOT, "annotation")
    DATA_DIR = os.path.join(DATASET_ROOT, "data")
    
    def __init__(self):
        """Initialize extractor."""
        self.datasets = {}  # {test_name: CocoDataset}
        self.all_images = []
        self.all_annotations = []
        
        logger.info("=" * 100)
        logger.info("DATASET EXTRACTION & PROCESSING")
        logger.info(f"Root: {self.DATASET_ROOT}")
        logger.info("=" * 100)
    
    def extract_all_json(self) -> Dict[str, CocoDataset]:
        """Extract all JSON annotations from dataset."""
        logger.info("\n[PHASE 1] EXTRACTING JSON ANNOTATIONS")
        logger.info("=" * 100)
        
        annotation_files = sorted(Path(self.ANNOTATION_DIR).rglob("*.json"))
        logger.info(f"Found {len(annotation_files)} annotation files")
        
        for annotation_file in annotation_files:
            test_name = annotation_file.parent.name
            
            try:
                with open(annotation_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Parse COCO format
                images_dict = {}
                categories = {}
                
                # Extract categories
                for cat in coco_data.get('categories', []):
                    categories[cat['id']] = cat['name']
                
                # Extract images
                for img_info in coco_data.get('images', []):
                    img_id = img_info['id']
                    images_dict[img_id] = ImageData(
                        image_id=img_id,
                        file_path=img_info['file_name'],
                        width=img_info['width'],
                        height=img_info['height'],
                        annotations=[]
                    )
                
                # Extract annotations
                total_anns = 0
                for ann in coco_data.get('annotations', []):
                    img_id = ann['image_id']
                    if img_id in images_dict:
                        images_dict[img_id].annotations.append({
                            'id': ann['id'],
                            'category': categories.get(ann['category_id'], 'unknown'),
                            'bbox': ann['bbox'],  # [x, y, width, height]
                            'area': ann.get('area', 0),
                        })
                        total_anns += 1
                
                # Create dataset
                dataset = CocoDataset(
                    images=images_dict,
                    categories=categories,
                    info=coco_data.get('info', {}),
                    total_annotations=total_anns
                )
                
                self.datasets[test_name] = dataset
                
                logger.info(f"\n✓ {test_name}")
                logger.info(f"  - Images: {len(images_dict)}")
                logger.info(f"  - Annotations: {total_anns}")
                logger.info(f"  - Categories: {list(categories.values())}")
                
                self.all_annotations.extend([a for anns in images_dict.values() for a in anns.annotations])
                self.all_images.extend(images_dict.values())
                
            except Exception as e:
                logger.error(f"Error extracting {annotation_file}: {e}")
        
        logger.info(f"\n[SUMMARY]")
        logger.info(f"  Tests extracted: {len(self.datasets)}")
        logger.info(f"  Total images: {len(self.all_images)}")
        logger.info(f"  Total annotations: {len(self.all_annotations)}")
        
        return self.datasets
    
    def load_images_for_test(self, test_name: str, limit: int = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """Load actual image arrays for a test."""
        logger.info(f"\n[LOADING IMAGES] {test_name}")
        
        if test_name not in self.datasets:
            logger.error(f"Test not found: {test_name}")
            return [], []
        
        dataset = self.datasets[test_name]
        images_loaded = []
        annotations_loaded = []
        
        # Find data directory
        test_data_dir = None
        for test_dir in Path(self.DATA_DIR).glob(f"*{test_name.split('_')[0]}*"):
            if test_dir.is_dir():
                # Find the specific subdirectory
                for subdir in test_dir.iterdir():
                    if subdir.is_dir():
                        test_data_dir = subdir
                        break
            if test_data_dir:
                break
        
        if not test_data_dir:
            logger.warning(f"Data directory not found for {test_name}")
            return [], []
        
        logger.info(f"  Data directory: {test_data_dir}")
        
        # Load images
        count = 0
        for img_data in dataset.images.values():
            if limit and count >= limit:
                break
            
            image_path = test_data_dir / img_data.file_path
            
            if image_path.exists():
                try:
                    img_array = cv2.imread(str(image_path))
                    if img_array is not None:
                        img_data.image_array = img_array
                        img_data.is_loaded = True
                        images_loaded.append(img_array)
                        annotations_loaded.append({
                            'image_id': img_data.image_id,
                            'annotations': img_data.annotations,
                            'shape': img_array.shape
                        })
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"  Loaded: {len(images_loaded)} images")
        return images_loaded, annotations_loaded
    
    def get_annotations_summary(self) -> Dict:
        """Generate summary of all annotations."""
        logger.info("\n[ANNOTATION SUMMARY]")
        logger.info("=" * 100)
        
        summary = {
            'total_annotations': len(self.all_annotations),
            'by_category': defaultdict(int),
            'bbox_statistics': {
                'widths': [],
                'heights': [],
                'areas': [],
            },
            'by_test': {}
        }
        
        for ann in self.all_annotations:
            # Category count
            category = ann.get('category', 'unknown')
            summary['by_category'][category] += 1
            
            # Bbox stats
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                summary['bbox_statistics']['widths'].append(bbox[2])
                summary['bbox_statistics']['heights'].append(bbox[3])
                summary['bbox_statistics']['areas'].append(bbox[2] * bbox[3])
        
        # Per-test statistics
        for test_name, dataset in self.datasets.items():
            summary['by_test'][test_name] = {
                'images': len(dataset.images),
                'annotations': dataset.total_annotations,
            }
        
        # Calculate stats
        if summary['bbox_statistics']['areas']:
            areas = summary['bbox_statistics']['areas']
            summary['bbox_statistics'] = {
                'min_area': min(areas),
                'max_area': max(areas),
                'mean_area': np.mean(areas),
                'median_area': np.median(areas),
                'std_area': np.std(areas),
                'min_width': min(summary['bbox_statistics']['widths']),
                'max_width': max(summary['bbox_statistics']['widths']),
                'min_height': min(summary['bbox_statistics']['heights']),
                'max_height': max(summary['bbox_statistics']['heights']),
            }
        
        logger.info(f"\nAnnotation Breakdown:")
        for category, count in summary['by_category'].items():
            logger.info(f"  {category}: {count}")
        
        logger.info(f"\nBounding Box Statistics:")
        for key, value in summary['bbox_statistics'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.1f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info(f"\nPer-Test Statistics:")
        for test_name, stats in summary['by_test'].items():
            logger.info(f"  {test_name}: {stats['images']} images, {stats['annotations']} annotations")
        
        return summary
    
    def export_to_coco_batch(self, output_dir: str, test_names: List[str] = None):
        """Export selected tests to separate COCO-format JSON files."""
        logger.info(f"\n[EXPORTING DATASETS] to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        tests_to_export = test_names if test_names else list(self.datasets.keys())
        
        for test_name in tests_to_export:
            if test_name not in self.datasets:
                logger.warning(f"Test not found: {test_name}")
                continue
            
            dataset = self.datasets[test_name]
            
            # Build COCO format
            coco_export = {
                'info': {
                    'description': f'HackHPI2026 - {test_name}',
                    'version': '1.0',
                    'year': 2023
                },
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': cat_id, 'name': cat_name}
                    for cat_id, cat_name in dataset.categories.items()
                ]
            }
            
            # Add images
            for img_id, img_data in dataset.images.items():
                coco_export['images'].append({
                    'id': img_data.image_id,
                    'file_name': img_data.file_path,
                    'width': img_data.width,
                    'height': img_data.height
                })
            
            # Add annotations
            ann_id = 0
            for img_data in dataset.images.values():
                for ann in img_data.annotations:
                    # Find category ID
                    cat_id = next((cid for cid, cname in dataset.categories.items() 
                                  if cname == ann['category']), 0)
                    
                    coco_export['annotations'].append({
                        'id': ann_id,
                        'image_id': img_data.image_id,
                        'category_id': cat_id,
                        'bbox': ann['bbox'],
                        'area': ann['area'],
                        'iscrowd': 0
                    })
                    ann_id += 1
            
            # Save
            output_file = os.path.join(output_dir, f"{test_name}_coco.json")
            with open(output_file, 'w') as f:
                json.dump(coco_export, f, indent=2)
            
            logger.info(f"✓ Exported: {output_file}")
    
    def create_train_val_split(self, output_dir: str, train_ratio: float = 0.8):
        """Create train/validation split from all data."""
        logger.info(f"\n[CREATING TRAIN/VAL SPLIT] ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% val)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all images and shuffle
        all_imgs = list(self.all_images)
        np.random.shuffle(all_imgs)
        
        split_idx = int(len(all_imgs) * train_ratio)
        train_imgs = all_imgs[:split_idx]
        val_imgs = all_imgs[split_idx:]
        
        # Create COCO format for each split
        for split_name, split_data in [('train', train_imgs), ('val', val_imgs)]:
            coco_split = {
                'info': {'description': f'HackHPI2026 {split_name} split', 'version': '1.0'},
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': 0, 'name': 'human_person'},
                    {'id': 1, 'name': 'human_manikin'}
                ]
            }
            
            # Collect unique annotations for this split
            ann_id = 0
            for img_data in split_data:
                coco_split['images'].append({
                    'id': img_data.image_id,
                    'file_name': img_data.file_path,
                    'width': img_data.width,
                    'height': img_data.height
                })
                
                for ann in img_data.annotations:
                    cat_id = 0 if 'person' in ann.get('category', '').lower() else 1
                    
                    coco_split['annotations'].append({
                        'id': ann_id,
                        'image_id': img_data.image_id,
                        'category_id': cat_id,
                        'bbox': ann['bbox'],
                        'area': ann['area'],
                        'iscrowd': 0
                    })
                    ann_id += 1
            
            # Save
            output_file = os.path.join(output_dir, f"hackhpi2026_{split_name}.json")
            with open(output_file, 'w') as f:
                json.dump(coco_split, f, indent=2)
            
            logger.info(f"✓ {split_name.upper()}: {len(split_data)} images, {len(coco_split['annotations'])} annotations")


def main():
    """Run dataset extraction and processing."""
    
    extractor = DatasetExtractor()
    
    # Phase 1: Extract JSON
    extractor.extract_all_json()
    
    # Phase 2: Get statistics
    summary = extractor.get_annotations_summary()
    
    # Phase 3: Load sample images
    if extractor.datasets:
        first_test = list(extractor.datasets.keys())[0]
        images, annotations = extractor.load_images_for_test(first_test, limit=5)
        logger.info(f"\nLoaded {len(images)} sample images")
    
    # Phase 4: Export datasets
    export_dir = os.path.join(log_dir, "dataset_exports")
    extractor.export_to_coco_batch(export_dir)
    
    # Phase 5: Create train/val split
    split_dir = os.path.join(log_dir, "train_val_split")
    extractor.create_train_val_split(split_dir, train_ratio=0.8)
    
    logger.info("\n" + "=" * 100)
    logger.info("✅ DATASET EXTRACTION COMPLETE")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
