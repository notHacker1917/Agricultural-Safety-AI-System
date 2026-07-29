import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Fine-tune on dataset
    # Lightweight: small epochs, batch, imgsz
    model.train(
        data='data.yaml',
        epochs=5,  # lightweight
        batch=4,   # small batch
        imgsz=640, # standard
        device=device,
        save=True,
        save_period=1,
        project='runs/train',
        name='person_finetune'
    )
    
    logging.info("Training completed")

if __name__ == "__main__":
    main()