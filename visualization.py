import cv2
import numpy as np
import logging

class Visualizer:
    """
    Visualize detections and trajectories.
    """
    def __init__(self, storage=None):
        self.storage = storage
        logging.info("Visualizer initialized")

    def get_risk_color(self, risk):
        """
        Get color based on risk level.

        Args:
            risk (float): Risk score 0-1.

        Returns:
            tuple: BGR color.
        """
        if risk < 0.3:
            return (0, 255, 0)  # green
        elif risk < 0.7:
            return (0, 255, 255)  # yellow
        else:
            return (0, 0, 255)  # red

    def draw(self, frame, tracked_objects, trajectories, risks):
        """
        Draw on frame.

        Args:
            frame (numpy array): Frame.
            tracked_objects (dict): {id: (bbox, mask, is_predicted, is_occluded, occlusion_duration)}
            trajectories (dict): {id: [(x,y), ...]}
            risks (dict): {id: risk}

        Returns:
            numpy array: Drawn frame.
        """
        for obj_id, (bbox, mask, is_predicted, is_occluded, occlusion_duration) in tracked_objects.items():
            risk = risks.get(obj_id, 0.0)
            color = self.get_risk_color(risk)
            
            # If occluded, use red color
            if is_occluded:
                color = (0, 0, 255)  # red for occluded
            elif is_predicted:
                color = (128, 128, 128)  # gray for predicted
            
            # Draw bbox with risk color
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Draw mask (bbox for simplicity)
            cv2.rectangle(frame, (int(mask[0]), int(mask[1])), (int(mask[2]), int(mask[3])), color, 1)
            
            # Draw trajectory as smooth polyline (last 30 frames)
            if obj_id in trajectories and len(trajectories[obj_id]) > 1:
                points = np.array(trajectories[obj_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
            
            # Get speed
            speed = 0
            if self.storage:
                velocity = self.storage.get_velocity(obj_id)
                speed = np.linalg.norm(velocity)
            
            # Label ID and speed
            label = f"ID: {obj_id} Speed: {speed:.1f}"
            if is_occluded:
                label += f" (Occluded {occlusion_duration}f)"
            elif is_predicted:
                label += " (Predicted)"
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw risk score
            text = f"Risk: {risk:.2f}"
            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # ALERT if risk > 0.7
            if risk > 0.7:
                cv2.putText(frame, "ALERT", (int(bbox[0]), int(bbox[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return frame

    def draw_overlays(self, frame, tracks, trajectory_storage, risk_results):
        """
        Draw overlays on frame for dashboard.

        Args:
            frame (numpy array): Frame.
            tracks (list): List of track dicts with 'id', 'bbox'.
            trajectory_storage: TrajectoryStorage instance.
            risk_results (list): List of risk dicts.

        Returns:
            numpy array: Annotated frame.
        """
        # Create risk dict from results
        risks = {r.get('id', i): r['risk_score'] for i, r in enumerate(risk_results)}
        
        # Create trajectories dict
        trajectories = {}
        for track in tracks:
            traj = trajectory_storage.get_trajectory(track['id'])
            if traj:
                trajectories[track['id']] = [(int(p[0]), int(p[1])) for p in traj[-30:]]  # Last 30 frames
        
        # Draw bounding boxes and IDs
        for track in tracks:
            bbox = track['bbox']
            obj_id = track['id']
            risk = risks.get(obj_id, 0.0)
            color = self.get_risk_color(risk)
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            
            # Draw trajectory
            if obj_id in trajectories and len(trajectories[obj_id]) > 1:
                points = np.array(trajectories[obj_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
            
            # Draw predicted future path (dotted line) - simplified
            if risk_results:
                for r in risk_results:
                    if r.get('id') == obj_id and 'predicted_path' in r:
                        pred_path = r['predicted_path']
                        if len(pred_path) > 1:
                            pred_points = np.array([(int(p[0]), int(p[1])) for p in pred_path], dtype=np.int32)
                            for i in range(len(pred_points) - 1):
                                cv2.line(frame, tuple(pred_points[i]), tuple(pred_points[i+1]), color, 1, cv2.LINE_AA)
                                # Dotted effect
                                if i % 2 == 0:
                                    cv2.line(frame, tuple(pred_points[i]), tuple(pred_points[i+1]), (255,255,255), 1, cv2.LINE_AA)
            
            # Label
            label = f"ID: {obj_id}"
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame