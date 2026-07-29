import json
import os
import numpy as np
import logging

class SafetyEngine:
    """
    Predictive safety engine for agricultural tractor-human interactions.

    The engine computes current and future collision risk, estimates time-to-collision,
    and logs critical events when a predicted path enters the tractor safety zone.
    """

    def __init__(
        self,
        tractor_center=(0.5, 0.0),
        homography=None,
        storage=None,
        image_size=(640, 480),
        prediction_frames=20,
        safety_zone_fraction=0.3,
        critical_ttc_threshold=10,
        critical_events_path='outputs/critical_events.json',
    ):
        self.tractor_center = np.array(tractor_center)
        self.homography = homography
        self.storage = storage
        self.width, self.height = image_size
        self.prediction_frames = prediction_frames
        self.safety_zone_fraction = safety_zone_fraction
        self.critical_ttc_threshold = critical_ttc_threshold
        self.critical_events_path = critical_events_path
        os.makedirs(os.path.dirname(self.critical_events_path), exist_ok=True)
        logging.info('Predictive safety engine initialized')

    def compute_velocity(self, trajectory, window=5):
        if trajectory is None or len(trajectory) < 2:
            return np.array([0.0, 0.0])
        points = np.array(trajectory[-window:])
        diffs = np.diff(points, axis=0)
        if len(diffs) == 0:
            return np.array([0.0, 0.0])
        return np.mean(diffs, axis=0)

    def predict_future_positions(self, centroid, velocity, n_steps=None):
        n_steps = n_steps or self.prediction_frames
        return [
            (centroid[0] + velocity[0] * step, centroid[1] + velocity[1] * step)
            for step in range(1, n_steps + 1)
        ]

    def predicted_safety_zone_entry(self, future_positions):
        threshold_y = self.height * (1.0 - self.safety_zone_fraction)
        for step_index, (x, y) in enumerate(future_positions, start=1):
            if y >= threshold_y:
                return step_index, (x, y)
        return None, None

    def risk_level_from_score(self, score):
        if score >= 0.7:
            return 'CRITICAL'
        if score >= 0.35:
            return 'WARNING'
        return 'SAFE'

    def log_critical_event(self, event):
        existing = []
        if os.path.exists(self.critical_events_path):
            try:
                with open(self.critical_events_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(event)
        with open(self.critical_events_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)

    def compute_risk(self, obj_id, bbox, trajectory, frame_index=None, current_risk=0.0):
        centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        bottom_y = bbox[3]

        distance_factor = np.clip(bottom_y / float(self.height), 0.0, 1.0)

        velocity = self.compute_velocity(trajectory)
        speed = np.linalg.norm(velocity)
        max_speed = 12.0
        velocity_factor = np.clip(speed / max_speed, 0.0, 1.0)

        tractor_point = np.array([self.width / 2.0, self.height])
        person_vector = tractor_point - np.array(centroid)
        person_norm = np.linalg.norm(person_vector)
        velocity_norm = np.linalg.norm(velocity)
        if person_norm > 0.0 and velocity_norm > 0.0:
            direction_score = np.dot(velocity, person_vector) / (velocity_norm * person_norm)
            direction_factor = np.clip(direction_score, 0.0, 1.0)
        else:
            direction_factor = 0.0

        future_positions = self.predict_future_positions(centroid, velocity)
        time_to_collision, collision_point = self.predicted_safety_zone_entry(future_positions)
        predicted_factor = 0.0
        if time_to_collision is not None:
            predicted_factor = 1.0 - min(time_to_collision / float(self.prediction_frames), 1.0)

        predicted_score = np.clip(
            0.35 * distance_factor
            + 0.2 * velocity_factor
            + 0.2 * direction_factor
            + 0.5 * predicted_factor,
            0.0,
            1.0,
        )

        combined_score = predicted_score
        if current_risk:
            combined_score = np.clip(0.4 * current_risk + 0.6 * predicted_score, 0.0, 1.0)

        risk_level = self.risk_level_from_score(combined_score)
        alert = risk_level == 'CRITICAL' and time_to_collision is not None and time_to_collision <= self.critical_ttc_threshold

        if alert:
            event = {
                'frame_index': int(frame_index) if frame_index is not None else None,
                'object_id': int(obj_id),
                'risk_score': float(combined_score),
                'time_to_collision': int(time_to_collision) if time_to_collision is not None else None,
                'risk_level': risk_level,
                'collision_point': [float(collision_point[0]), float(collision_point[1])] if collision_point is not None else None,
                'bbox': [float(x) for x in bbox],
            }
            try:
                self.log_critical_event(event)
            except Exception as e:
                logging.warning(f'Failed to log critical event: {e}')

        return {
            'risk_score': float(combined_score),
            'time_to_collision': int(time_to_collision) if time_to_collision is not None else None,
            'risk_level': risk_level,
            'alert': bool(alert),
            'predicted_path': [[float(x), float(y)] for x, y in future_positions],
            'collision_point': [float(collision_point[0]), float(collision_point[1])] if collision_point is not None else None,
        }
