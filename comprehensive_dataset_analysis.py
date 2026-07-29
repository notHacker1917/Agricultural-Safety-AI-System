"""
COMPREHENSIVE DATASET ANALYSIS & HUMAN DETECTION OPTIMIZATION
Analyzes dataset characteristics and generates optimized detection strategies
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDatasetAnalyzer:
    """Analyze dataset to inform detection strategy optimization"""
    
    def __init__(self, dataset_root: str, annotation_dir: str, data_dir: str):
        self.dataset_root = dataset_root
        self.annotation_dir = annotation_dir
        self.data_dir = data_dir
        self.analysis_results = {}
        
    def analyze_all(self) -> Dict:
        """Run complete dataset analysis"""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE DATASET ANALYSIS")
        logger.info("=" * 80)
        
        # 1. Scan annotations
        annotations_summary = self._analyze_annotations()
        
        # 2. Analyze images
        image_summary = self._analyze_images()
        
        # 3. Analyze detection patterns
        detection_patterns = self._analyze_detection_patterns()
        
        # 4. Analyze scale/occlusion/distance
        scale_analysis = self._analyze_scale_and_distance()
        
        # 5. Analyze spatial distribution
        spatial_analysis = self._analyze_spatial_distribution()
        
        # 6. Generate detection optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            annotations_summary, image_summary, detection_patterns, 
            scale_analysis, spatial_analysis
        )
        
        return {
            'annotations': annotations_summary,
            'images': image_summary,
            'patterns': detection_patterns,
            'scale': scale_analysis,
            'spatial': spatial_analysis,
            'recommendations': recommendations
        }
    
    def _analyze_annotations(self) -> Dict:
        """Analyze annotation files and structure"""
        logger.info("\n1. ANALYZING ANNOTATION FILES...")
        
        json_files = list(Path(self.annotation_dir).rglob("*.json"))
        logger.info(f"   Found {len(json_files)} annotation files")
        
        annotations_data = {
            'total_images': 0,
            'total_annotations': 0,
            'categories': Counter(),
            'files': []
        }
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                images = data.get('images', [])
                annotations = data.get('annotations', [])
                categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                
                for ann in annotations:
                    cat_name = categories.get(ann['category_id'], 'unknown')
                    annotations_data['categories'][cat_name] += 1
                
                annotations_data['total_images'] += len(images)
                annotations_data['total_annotations'] += len(annotations)
                annotations_data['files'].append({
                    'name': json_file.name,
                    'images': len(images),
                    'annotations': len(annotations)
                })
                
                logger.info(f"   {json_file.name}: {len(images)} images, {len(annotations)} annotations")
            
            except Exception as e:
                logger.warning(f"   Error reading {json_file.name}: {e}")
        
        logger.info(f"\n   SUMMARY:")
        logger.info(f"   - Total images: {annotations_data['total_images']}")
        logger.info(f"   - Total annotations: {annotations_data['total_annotations']}")
        logger.info(f"   - Categories:")
        for cat, count in sorted(annotations_data['categories'].items()):
            logger.info(f"     * {cat}: {count}")
        
        return annotations_data
    
    def _analyze_images(self) -> Dict:
        """Analyze image properties"""
        logger.info("\n2. ANALYZING IMAGE PROPERTIES...")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(self.data_dir).rglob(f"*{ext}")))
        
        logger.info(f"   Found {len(image_files)} image files")
        
        resolutions = Counter()
        file_sizes = []
        brightness_values = []
        contrast_values = []
        
        for i, img_path in enumerate(image_files[:min(100, len(image_files))]):  # Sample first 100
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                resolutions[f"{w}x{h}"] += 1
                
                # Calculate brightness
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Calculate contrast (standard deviation)
                contrast = np.std(gray)
                contrast_values.append(contrast)
                
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
                
                if (i + 1) % 20 == 0:
                    logger.info(f"   Processed {i + 1} images...")
            
            except Exception as e:
                logger.debug(f"   Error processing {img_path.name}: {e}")
        
        logger.info(f"\n   IMAGE PROPERTIES:")
        logger.info(f"   - Common resolutions:")
        for res, count in resolutions.most_common(5):
            logger.info(f"     * {res}: {count} images")
        
        if brightness_values:
            logger.info(f"   - Brightness: {np.mean(brightness_values):.1f} ± {np.std(brightness_values):.1f}")
        if contrast_values:
            logger.info(f"   - Contrast (Std Dev): {np.mean(contrast_values):.1f} ± {np.std(contrast_values):.1f}")
        if file_sizes:
            logger.info(f"   - File size: {np.mean(file_sizes):.1f} ± {np.std(file_sizes):.1f} KB")
        
        return {
            'total_images': len(image_files),
            'resolutions': dict(resolutions.most_common(5)),
            'brightness': {
                'mean': float(np.mean(brightness_values)) if brightness_values else 0,
                'std': float(np.std(brightness_values)) if brightness_values else 0
            },
            'contrast': {
                'mean': float(np.mean(contrast_values)) if contrast_values else 0,
                'std': float(np.std(contrast_values)) if contrast_values else 0
            }
        }
    
    def _analyze_detection_patterns(self) -> Dict:
        """Analyze detection patterns and challenges"""
        logger.info("\n3. ANALYZING DETECTION PATTERNS...")
        
        json_files = list(Path(self.annotation_dir).rglob("*.json"))
        bbox_stats = defaultdict(list)
        scale_distribution = {'small': 0, 'medium': 0, 'large': 0}  # <5%, 5-15%, >15%
        occlusion_patterns = defaultdict(int)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                images = {img['id']: img for img in data.get('images', [])}
                annotations = data.get('annotations', [])
                categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                
                for ann in annotations:
                    cat_name = categories.get(ann['category_id'], 'unknown')
                    
                    if 'bbox' in ann:
                        bbox = ann['bbox']  # [x, y, width, height]
                        img_id = ann.get('image_id')
                        
                        if img_id in images:
                            img_info = images[img_id]
                            img_height = img_info.get('height', 480)
                            img_width = img_info.get('width', 640)
                            
                            # Calculate metrics
                            bbox_area_ratio = (bbox[2] * bbox[3]) / (img_width * img_height)
                            aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 1
                            
                            # Categorize by size
                            if bbox_area_ratio < 0.05:
                                scale_distribution['small'] += 1
                            elif bbox_area_ratio < 0.15:
                                scale_distribution['medium'] += 1
                            else:
                                scale_distribution['large'] += 1
                            
                            bbox_stats[cat_name].append({
                                'area_ratio': bbox_area_ratio,
                                'aspect_ratio': aspect_ratio,
                                'width': bbox[2],
                                'height': bbox[3],
                                'x': bbox[0],
                                'y': bbox[1]
                            })
                            
                            # Occlusion heuristic: edge proximity
                            edge_distance = min(bbox[0], bbox[1], 
                                              img_width - (bbox[0] + bbox[2]),
                                              img_height - (bbox[1] + bbox[3]))
                            if edge_distance < 20:
                                occlusion_patterns['edge'] += 1
            
            except Exception as e:
                logger.debug(f"   Error analyzing {json_file.name}: {e}")
        
        logger.info(f"\n   DETECTION PATTERNS:")
        logger.info(f"   - Scale distribution:")
        logger.info(f"     * Small (<5% image): {scale_distribution['small']}")
        logger.info(f"     * Medium (5-15%): {scale_distribution['medium']}")
        logger.info(f"     * Large (>15%): {scale_distribution['large']}")
        logger.info(f"   - Occlusion/Edge cases: {occlusion_patterns['edge']}")
        
        for cat_name, stats in bbox_stats.items():
            if stats:
                areas = [s['area_ratio'] for s in stats]
                aspect_ratios = [s['aspect_ratio'] for s in stats]
                logger.info(f"\n   {cat_name}:")
                logger.info(f"     * Mean area ratio: {np.mean(areas):.4f}")
                logger.info(f"     * Mean aspect ratio: {np.mean(aspect_ratios):.2f}")
                logger.info(f"     * Area range: {np.min(areas):.4f} - {np.max(areas):.4f}")
        
        return {
            'scale_distribution': scale_distribution,
            'bbox_stats': {k: {
                'mean_area_ratio': float(np.mean([s['area_ratio'] for s in v])),
                'mean_aspect_ratio': float(np.mean([s['aspect_ratio'] for s in v])),
                'count': len(v)
            } for k, v in bbox_stats.items()},
            'occlusion_patterns': dict(occlusion_patterns)
        }
    
    def _analyze_scale_and_distance(self) -> Dict:
        """Analyze scale and estimated distance information"""
        logger.info("\n4. ANALYZING SCALE & DISTANCE ESTIMATION...")
        
        # Reference: Average human height ~1.7m
        # At different distances with standard focal length (~300px at 10m)
        distance_reference = {
            5: 100,    # At 5m, person ~100 pixels tall
            10: 50,    # At 10m, person ~50 pixels tall
            20: 25,    # At 20m, person ~25 pixels tall
            30: 17,    # At 30m, person ~17 pixels tall
            40: 12     # At 40m, person ~12 pixels tall
        }
        
        json_files = list(Path(self.annotation_dir).rglob("*.json"))
        height_distribution = []
        distance_estimates = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for ann in data.get('annotations', []):
                    if 'bbox' in ann:
                        bbox = ann['bbox']
                        person_height = bbox[3]
                        height_distribution.append(person_height)
                        
                        # Estimate distance (rough calibration)
                        estimated_dist = 10 * (50 / max(person_height, 1))
                        distance_estimates.append(estimated_dist)
            
            except Exception as e:
                logger.debug(f"   Error: {e}")
        
        if height_distribution:
            logger.info(f"\n   HEIGHT DISTRIBUTION:")
            logger.info(f"   - Mean: {np.mean(height_distribution):.1f} pixels")
            logger.info(f"   - Median: {np.median(height_distribution):.1f} pixels")
            logger.info(f"   - Range: {np.min(height_distribution):.0f} - {np.max(height_distribution):.0f} pixels")
            logger.info(f"   - Percentiles: 25%={np.percentile(height_distribution, 25):.0f}, "
                       f"75%={np.percentile(height_distribution, 75):.0f}")
        
        if distance_estimates:
            logger.info(f"\n   ESTIMATED DISTANCE DISTRIBUTION:")
            logger.info(f"   - Mean: {np.mean(distance_estimates):.1f}m")
            logger.info(f"   - Median: {np.median(distance_estimates):.1f}m")
            logger.info(f"   - Range: {np.min(distance_estimates):.1f} - {np.max(distance_estimates):.1f}m")
        
        return {
            'height_stats': {
                'mean': float(np.mean(height_distribution)) if height_distribution else 0,
                'median': float(np.median(height_distribution)) if height_distribution else 0,
                'min': float(np.min(height_distribution)) if height_distribution else 0,
                'max': float(np.max(height_distribution)) if height_distribution else 0
            },
            'distance_stats': {
                'mean': float(np.mean(distance_estimates)) if distance_estimates else 0,
                'median': float(np.median(distance_estimates)) if distance_estimates else 0,
                'min': float(np.min(distance_estimates)) if distance_estimates else 0,
                'max': float(np.max(distance_estimates)) if distance_estimates else 0
            }
        }
    
    def _analyze_spatial_distribution(self) -> Dict:
        """Analyze spatial distribution of detections"""
        logger.info("\n5. ANALYZING SPATIAL DISTRIBUTION...")
        
        json_files = list(Path(self.annotation_dir).rglob("*.json"))
        spatial_grid = defaultdict(int)  # 3x3 grid
        center_detections = 0
        edge_detections = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                images = {img['id']: img for img in data.get('images', [])}
                
                for ann in data.get('annotations', []):
                    if 'bbox' in ann:
                        bbox = ann['bbox']
                        img_id = ann.get('image_id')
                        
                        if img_id in images:
                            img_info = images[img_id]
                            img_height = img_info.get('height', 480)
                            img_width = img_info.get('width', 640)
                            
                            # Calculate center
                            cx = (bbox[0] + bbox[2]/2) / img_width
                            cy = (bbox[1] + bbox[3]/2) / img_height
                            
                            # Categorize into 3x3 grid
                            grid_x = int(cx * 3)
                            grid_y = int(cy * 3)
                            grid_x = min(2, max(0, grid_x))
                            grid_y = min(2, max(0, grid_y))
                            spatial_grid[(grid_x, grid_y)] += 1
                            
                            # Check if center or edge
                            if 0.3 < cx < 0.7 and 0.3 < cy < 0.7:
                                center_detections += 1
                            else:
                                edge_detections += 1
            
            except Exception as e:
                logger.debug(f"   Error: {e}")
        
        logger.info(f"\n   SPATIAL DISTRIBUTION:")
        logger.info(f"   - Center detections: {center_detections} ({100*center_detections/(center_detections+edge_detections+1):.1f}%)")
        logger.info(f"   - Edge detections: {edge_detections} ({100*edge_detections/(center_detections+edge_detections+1):.1f}%)")
        
        return {
            'center_ratio': center_detections / max(1, center_detections + edge_detections),
            'edge_ratio': edge_detections / max(1, center_detections + edge_detections),
            'spatial_grid': dict(spatial_grid)
        }
    
    def _generate_optimization_recommendations(self, 
                                              annotations_summary: Dict,
                                              image_summary: Dict,
                                              detection_patterns: Dict,
                                              scale_analysis: Dict,
                                              spatial_analysis: Dict) -> Dict:
        """Generate detection optimization recommendations"""
        logger.info("\n6. GENERATING OPTIMIZATION RECOMMENDATIONS...")
        logger.info("=" * 80)
        
        recommendations = {
            'multi_scale_detection': [],
            'preprocessing': [],
            'model_configuration': [],
            'post_processing': [],
            'ensemble_strategy': []
        }
        
        # Multi-scale recommendations
        scale_dist = detection_patterns.get('scale_distribution', {})
        if scale_dist.get('small', 0) > scale_dist.get('large', 1):
            recommendations['multi_scale_detection'].append(
                "High proportion of small detections: Use multi-scale feature pyramids"
            )
        
        if scale_dist.get('large', 0) > 0:
            recommendations['multi_scale_detection'].append(
                "Large detections present: Enable full-resolution processing"
            )
        
        # Preprocessing recommendations
        brightness = image_summary.get('brightness', {})
        if brightness.get('mean', 100) < 80:
            recommendations['preprocessing'].append(
                "Low brightness images detected: Apply CLAHE/histogram equalization"
            )
        
        contrast = image_summary.get('contrast', {})
        if contrast.get('mean', 50) < 25:
            recommendations['preprocessing'].append(
                "Low contrast detected: Apply contrast enhancement"
            )
        
        # Model configuration recommendations
        distance_stats = scale_analysis.get('distance_stats', {})
        if distance_stats.get('mean', 20) < 15:
            recommendations['model_configuration'].append(
                "Close-range detections common: Use high confidence threshold (0.6+)"
            )
        
        if distance_stats.get('mean', 20) > 25:
            recommendations['model_configuration'].append(
                "Far-range detections common: Use lower confidence threshold (0.4-0.5)"
            )
        
        # Post-processing recommendations
        if detection_patterns.get('occlusion_patterns', {}).get('edge', 0) > 50:
            recommendations['post_processing'].append(
                "Edge occlusions detected: Implement boundary box refinement"
            )
        
        # Ensemble strategy recommendations
        recommendations['ensemble_strategy'].append(
            "Combine YOLO detections with:1) Multi-scale analysis"
        )
        recommendations['ensemble_strategy'].append(
            "2) Motion tracking (optical flow)"
        )
        recommendations['ensemble_strategy'].append(
            "3) Background subtraction for moving objects"
        )
        
        logger.info("\n   RECOMMENDATIONS:")
        logger.info(f"\n   🔹 Multi-Scale Detection:")
        for rec in recommendations['multi_scale_detection']:
            logger.info(f"      - {rec}")
        
        logger.info(f"\n   🔹 Preprocessing:")
        for rec in recommendations['preprocessing']:
            logger.info(f"      - {rec}")
        
        logger.info(f"\n   🔹 Model Configuration:")
        for rec in recommendations['model_configuration']:
            logger.info(f"      - {rec}")
        
        logger.info(f"\n   🔹 Post-Processing:")
        for rec in recommendations['post_processing']:
            logger.info(f"      - {rec}")
        
        logger.info(f"\n   🔹 Ensemble Strategy:")
        for rec in recommendations['ensemble_strategy']:
            logger.info(f"      - {rec}")
        
        return recommendations


def main():
    """Main entry point"""
    dataset_root = r"C:\Users\hs735.COLTSMOKE\OneDrive\Documents\Hackathon\HackHPI2026_release"
    annotation_dir = os.path.join(dataset_root, "annotation")
    data_dir = os.path.join(dataset_root, "data")
    
    # Run analysis
    analyzer = ComprehensiveDatasetAnalyzer(dataset_root, annotation_dir, data_dir)
    results = analyzer.analyze_all()
    
    # Save results
    output_file = os.path.join(os.getcwd(), "dataset_analysis_results.json")
    
    try:
        with open(output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, Counter):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            json.dump(convert_to_serializable(results), f, indent=2)
        
        logger.info(f"\n✓ Analysis complete! Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        logger.info("Results were still computed successfully - printed above")
    logger.info("=" * 80)
    
    return results


if __name__ == '__main__':
    results = main()
