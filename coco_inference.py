import os
import cv2
import json
import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)

def load_coco_dataset(annotations_path, images_dir):
    """
    Load COCO dataset annotations and images.

    Args:
        annotations_path (str): Path to COCO annotations JSON.
        images_dir (str): Directory containing images.

    Returns:
        list: List of (image_path, image_info)
    """
    with open(annotations_path, 'r') as f:
        coco = json.load(f)
    
    images = []
    for img_info in coco['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        if os.path.exists(img_path):
            images.append((img_path, img_info))
        else:
            logging.warning(f"Image not found: {img_path}")
    
    logging.info(f"Loaded {len(images)} images from COCO dataset")
    return images

def run_inference_and_visualize(images, model, output_dir):
    """
    Run YOLO inference, visualize bboxes, and save outputs.

    Args:
        images (list): List of (image_path, image_info)
        model: YOLO model
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, img_info in images:
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            continue
        
        try:
            results = model(img, classes=[0], verbose=False)  # person class
            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(img, f"{conf:.2f}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logging.warning(f"Inference failed for {img_path}: {e}")
        
        out_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(out_path, img)
        logging.info(f"Saved output to {out_path}")

def main():
    annotations_path = 'data/annotations.json'
    images_dir = 'data/images'
    output_dir = 'output'
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = YOLO('yolov8n.pt')
        model.to(device)
        logging.info(f"Model loaded on {device}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Load dataset
    images = load_coco_dataset(annotations_path, images_dir)
    if not images:
        logging.error("No images loaded")
        return
    
    # Run inference
    run_inference_and_visualize(images, model, output_dir)

if __name__ == "__main__":
    main()