import json
import logging
import os
import cv2
from coco_loader import COCODataset
from detection import ObjectDetector
from segmentation_tracking import DeepSORTTracker
from trajectory_storage import TrajectoryStorage
from safety_engine import SafetyEngine
from visualization import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load COCO dataset
    coco_dataset = COCODataset(
        'data/annotations/instances_val2017.json',
        'data/val2017'
    )

    # Initialize modules
    detector = ObjectDetector()
    tracker = DeepSORTTracker()
    storage = TrajectoryStorage()
    safety_engine = SafetyEngine()
    visualizer = Visualizer()

    frame_results = []

    # Process each image in the dataset
    for image_id in coco_dataset.get_image_ids():
        try:
            # Load image
            image = coco_dataset.get_image(image_id)
            frame_index = image_id  # Use image_id as frame_index

            # Detection
            detections = detector.detect(image)

            # Tracking
            tracks_dict = tracker.update(detections, image)

            # Update trajectories and compute risks
            risk_results = []
            tracks = []
            for obj_id, (bbox, mask, is_pred, is_occ, occ_dur) in tracks_dict.items():
                storage.update(frame_index, obj_id, bbox)
                trajectory = storage.get_trajectory(obj_id)
                risk = safety_engine.compute_risk(obj_id, bbox, trajectory, frame_index)
                risk['id'] = obj_id
                risk_results.append(risk)
                tracks.append({'id': obj_id, 'bbox': bbox})

            # Annotate frame
            annotated_frame = visualizer.draw_overlays(image, tracks, storage, risk_results)

            # Collect results for this frame
            frame_result = {
                'frame_index': frame_index,
                'detections': [{'bbox': bbox.tolist(), 'confidence': conf} for bbox, conf in detections],
                'tracks': [{'id': t['id'], 'bbox': t['bbox']} for t in tracks],
                'risks': risk_results,
                'annotated_image_path': f'outputs/annotated_frames/frame_{frame_index}.jpg'
            }
            frame_results.append(frame_result)

            # Save annotated image
            os.makedirs('outputs/annotated_frames', exist_ok=True)
            cv2.imwrite(frame_result['annotated_image_path'], annotated_frame)

            logging.info(f"Processed frame {frame_index}")

        except Exception as e:
            logging.error(f"Error processing image {image_id}: {e}")
            continue

    # Save results to JSON
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/frame_results.json', 'w') as f:
        json.dump(frame_results, f, indent=2)

    logging.info("Processing complete. Results saved to outputs/frame_results.json")

if __name__ == '__main__':
    main()