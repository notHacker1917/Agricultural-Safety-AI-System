#!/usr/bin/env python3
"""
Agricultural YOLO Training Script
Fine-tunes YOLO model on agricultural safety dataset
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgriYOLOTrainer:
    def __init__(self, data_yaml='data/data_agri.yaml', model_size='n'):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_base_model(self):
        """Load base YOLO model"""
        model_path = f'yolov8{self.model_size}.pt'
        if not Path(model_path).exists():
            logging.info(f"Downloading YOLOv8-{self.model_size} model...")
            model = YOLO(f'yolov8{self.model_size}.yaml')
        else:
            logging.info(f"Loading existing YOLOv8-{self.model_size} model")
            model = YOLO(model_path)

        return model

    def train_unified(self, epochs=50, batch_size=16, img_size=640):
        """Train on unified dataset"""
        logging.info("Starting unified agricultural training")

        model = self.load_base_model()

        # Create models directory
        models_dir = Path('data/models')
        models_dir.mkdir(exist_ok=True)

        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            save=True,
            save_period=10,
            project=str(models_dir),
            name=f'yolov8_agri_{self.model_size}',
            # Agricultural-specific augmentations
            augment=True,
            hsv_h=0.015,  # Hue augmentation for different lighting
            hsv_s=0.7,    # Saturation for weather conditions
            hsv_v=0.4,    # Brightness for time of day
            degrees=10,   # Rotation for different camera angles
            translate=0.1, # Translation for movement
            scale=0.5,    # Scale for distance variations
            shear=2,      # Shear for perspective changes
            perspective=0.0001,  # Perspective for terrain
            flipud=0.0,   # No vertical flip (people don't stand on heads)
            fliplr=0.5,   # Horizontal flip OK
            mosaic=1.0,   # Mosaic augmentation
            mixup=0.1,    # Mixup for robustness
        )

        # Save the best model
        best_model_path = models_dir / f'yolov8_agri_{self.model_size}.pt'
        model.save(str(best_model_path))

        logging.info(f"Training completed. Best model saved to: {best_model_path}")
        return results

    def train_environment_specific(self, env_name, epochs=30):
        """Train model specific to an environment"""
        logging.info(f"Training environment-specific model for: {env_name}")

        # Create environment-specific data yaml
        env_data_yaml = f'data/processed/{env_name}/data.yaml'
        env_yaml_content = f"""path: data/processed/{env_name}
train: images
val: images
nc: 1
names: ['person']
"""

        with open(env_data_yaml, 'w') as f:
            f.write(env_yaml_content)

        model = self.load_base_model()

        models_dir = Path('data/models')
        results = model.train(
            data=env_data_yaml,
            epochs=epochs,
            batch=8,  # Smaller batch for environment-specific
            imgsz=640,
            device=self.device,
            save=True,
            project=str(models_dir),
            name=f'yolov8_agri_{env_name}_{self.model_size}',
        )

        # Save model
        model_path = models_dir / f'yolov8_agri_{env_name}_{self.model_size}.pt'
        model.save(str(model_path))

        logging.info(f"Environment training completed: {model_path}")
        return results

    def validate_model(self, model_path, data_yaml=None):
        """Validate trained model"""
        if data_yaml is None:
            data_yaml = self.data_yaml

        logging.info(f"Validating model: {model_path}")

        model = YOLO(model_path)
        results = model.val(data=data_yaml, device=self.device)

        logging.info("Validation Results:")
        logging.info(f"  mAP50: {results.box.map50:.4f}")
        logging.info(f"  mAP50-95: {results.box.map:.4f}")
        logging.info(f"  Precision: {results.box.mp:.4f}")
        logging.info(f"  Recall: {results.box.mr:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Train Agricultural YOLO Model')
    parser.add_argument('--data-yaml', default='data/data_agri.yaml', help='Data configuration file')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--env-specific', help='Train environment-specific model')
    parser.add_argument('--validate-only', help='Only validate existing model')
    parser.add_argument('--data-root', default='data', help='Data root directory for custom setups')

    args = parser.parse_args()

    trainer = AgriYOLOTrainer(args.data_yaml, args.model_size)
    
    # If custom data root, update paths
    if args.data_root != 'data':
        trainer.data_yaml = f'{args.data_root}/data_agri.yaml'

    if args.validate_only:
        trainer.validate_model(args.validate_only)
    elif args.env_specific:
        trainer.train_environment_specific(args.env_specific, args.epochs)
    else:
        trainer.train_unified(args.epochs, args.batch_size)


if __name__ == '__main__':
    main()