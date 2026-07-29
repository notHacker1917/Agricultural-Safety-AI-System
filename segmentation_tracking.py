import numpy as np
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    """
    Tracking using DeepSORT for consistent IDs.
    """
    def __init__(self, max_predict_frames=5, min_speed=1.0):
        self.max_predict_frames = max_predict_frames
        self.min_speed = min_speed
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)
        logging.info("DeepSORT tracker initialized")

    def update(self, detections, frame=None):
        """
        Update tracker with new detections.

        Args:
            detections (list): List of (bbox, conf)
            frame (numpy array): Frame for embedding.

        Returns:
            dict: {id: (bbox, mask, is_predicted, is_occluded, occlusion_duration)}
        """
        # detections is already [(bbox, conf), ...]
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        tracked = {}
        for track in tracks:
            if track.is_confirmed() or (not track.is_deleted() and track.time_since_update <= self.max_predict_frames):
                # Calculate speed from Kalman filter state
                vx, vy = track.mean[2], track.mean[3]
                speed = np.sqrt(vx**2 + vy**2)
                is_predicted = not track.is_confirmed() or track.time_since_update > 0
                
                # Filter static objects unless predicted (they were moving)
                if is_predicted or speed > self.min_speed:
                    bbox = track.to_tlbr()  # x1,y1,x2,y2 (predicted if lost)
                    mask = self.bbox_to_mask(bbox)
                    is_occluded = track.time_since_update > 0
                    occlusion_duration = track.time_since_update if is_occluded else 0
                    tracked[track.track_id] = (bbox, mask, is_predicted, is_occluded, occlusion_duration)
        
        logging.debug(f"Tracked {len(tracked)} objects")
        return tracked

    def bbox_to_mask(self, bbox):
        """
        Mock mask from bbox.
        """
        return bbox