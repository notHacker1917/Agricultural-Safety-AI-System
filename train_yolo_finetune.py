import argparse
import json
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PERSON_CLASS_ID = 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_coco_annotations(annotations_path):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yolo_labels(label_path, bboxes, image_width, image_height):
    lines = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height
        lines.append(f"{PERSON_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def random_blur(image):
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def random_brightness(image):
    factor = random.uniform(0.6, 1.4)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def random_occlusion(image, bbox):
    occluded = image.copy()
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return occluded
    patch_width = max(1, int(w * random.uniform(0.2, 0.6)))
    patch_height = max(1, int(h * random.uniform(0.2, 0.6)))
    left = random.randint(x, max(x, x + w - patch_width))
    top = random.randint(y, max(y, y + h - patch_height))
    color = tuple(int(c) for c in np.random.randint(0, 80, size=3))
    cv2.rectangle(occluded, (left, top), (left + patch_width, top + patch_height), color, -1)
    return occluded


def apply_random_augmentation(image, bboxes):
    aug_image = image.copy()
    transforms = [random_blur, random_brightness]
    if bboxes:
        transforms.append(lambda img: random_occlusion(img, random.choice(bboxes)))
    random.shuffle(transforms)
    for transform in transforms[: random.randint(1, 2)]:
        aug_image = transform(aug_image)
    return aug_image


def split_train_val(image_ids, val_ratio, seed=42):
    random.seed(seed)
    ids = list(image_ids)
    random.shuffle(ids)
    split = int(len(ids) * (1.0 - val_ratio))
    return ids[:split], ids[split:]


def prepare_coco_dataset(annotations_path, images_dir, output_dir, val_ratio=0.2, augment_per_image=1):
    coco = load_coco_annotations(annotations_path)
    person_cat_id = None
    for cat in coco.get('categories', []):
        if cat.get('name') == 'person':
            person_cat_id = cat.get('id')
            break
    if person_cat_id is None:
        raise ValueError("COCO annotations do not contain a 'person' category")

    images_by_id = {img['id']: img for img in coco.get('images', [])}
    person_annots = [ann for ann in coco.get('annotations', []) if ann.get('category_id') == person_cat_id]
    image_to_bboxes = {}
    for ann in person_annots:
        if ann['image_id'] not in images_by_id:
            continue
        image_to_bboxes.setdefault(ann['image_id'], []).append(ann['bbox'])

    train_ids, val_ids = split_train_val(image_to_bboxes.keys(), val_ratio)

    train_images_dir = ensure_dir(os.path.join(output_dir, 'images', 'train'))
    val_images_dir = ensure_dir(os.path.join(output_dir, 'images', 'val'))
    train_labels_dir = ensure_dir(os.path.join(output_dir, 'labels', 'train'))
    val_labels_dir = ensure_dir(os.path.join(output_dir, 'labels', 'val'))

    def write_image_and_labels(image_id, split_dir, label_dir, suffix='', augment=False):
        img_info = images_by_id[image_id]
        src_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(src_path):
            logging.warning(f"Missing image: {src_path}")
            return
        image = cv2.imread(src_path)
        if image is None:
            logging.warning(f"Failed to load image: {src_path}")
            return
        if augment:
            image = apply_random_augmentation(image, image_to_bboxes[image_id])
            out_name = f"{Path(img_info['file_name']).stem}_aug{suffix}{Path(img_info['file_name']).suffix}"
        else:
            out_name = img_info['file_name']
        out_path = os.path.join(split_dir, out_name)
        cv2.imwrite(out_path, image)
        label_path = os.path.join(label_dir, Path(out_name).with_suffix('.txt'))
        save_yolo_labels(label_path, image_to_bboxes[image_id], image.shape[1], image.shape[0])

    for image_id in train_ids:
        write_image_and_labels(image_id, train_images_dir, train_labels_dir)
        for aug_idx in range(augment_per_image):
            write_image_and_labels(image_id, train_images_dir, train_labels_dir, suffix=aug_idx + 1, augment=True)

    for image_id in val_ids:
        write_image_and_labels(image_id, val_images_dir, val_labels_dir)

    return create_data_yaml(output_dir)


def create_data_yaml(output_dir):
    yaml_data = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['person'],
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        json.dump(yaml_data, f, indent=2)
    return yaml_path


def train_finetune(data_yaml, epochs, batch, imgsz, project, name, device):
    model = YOLO('yolov8n.pt')
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        augment=True,
        save=True,
        save_period=1,
        project=project,
        name=name,
        patience=10,
        lr0=0.01,
        workers=4,
    )
    return os.path.join(project, name)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv8 on agricultural COCO person dataset')
    parser.add_argument('--annotations', default='data/annotations.json', help='COCO annotations JSON path')
    parser.add_argument('--images', default='data/images', help='Directory containing COCO images')
    parser.add_argument('--output', default='data_finetune', help='Directory to store prepared dataset and output YAML')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--augment_per_image', type=int, default=1, help='Number of augmented copies per train image')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Training batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--project', default='runs/train', help='YOLO project directory')
    parser.add_argument('--name', default='agri_person_finetune', help='YOLO run name')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device: {device}')

    data_yaml = prepare_coco_dataset(
        annotations_path=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        augment_per_image=args.augment_per_image,
    )
    logging.info(f'Prepared finetuning dataset at {args.output}')

    output_path = train_finetune(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=device,
    )
    logging.info(f'Training complete. Results saved to {output_path}')
    logging.info('Review YOLO training curves in the run directory (plots and results files)')


if __name__ == '__main__':
    main()
