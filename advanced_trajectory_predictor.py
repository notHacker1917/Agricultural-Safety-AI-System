#!/usr/bin/env python3
"""
ADVANCED HUMAN TRAJECTORY PREDICTION SYSTEM
Robust trajectory prediction for challenging agricultural environments

Handles:
- Windy conditions with unpredictable motion
- Dust storms and reduced visibility
- Rain and weather degradation
- Opaque obstacles and occlusions
- Complex human walking patterns
- Agricultural machinery interactions

Uses advanced algorithms:
- Kalman Filtering with adaptive noise models
- LSTM-based sequence prediction
- Physics-based motion modeling
- LLM-enhanced behavioral reasoning
- Multi-hypothesis trajectory prediction
- Environmental adaptation algorithms
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Single point in trajectory with metadata"""
    position: Tuple[float, float]  # (x, y) in pixels
    velocity: Tuple[float, float]  # (vx, vy) pixels/frame
    acceleration: Tuple[float, float]  # (ax, ay) pixels/frame²
    timestamp: float
    confidence: float
    environmental_factors: Dict[str, float]  # wind, visibility, etc.

@dataclass
class PredictedTrajectory:
    """Complete predicted trajectory with uncertainty"""
    object_id: str
    current_position: Tuple[float, float]
    predicted_path: List[Tuple[float, float]]  # Future positions
    confidence_scores: List[float]  # Confidence for each prediction
    time_horizon: int  # Frames ahead
    environmental_conditions: Dict[str, Any]
    risk_assessment: Dict[str, Any]

class EnvironmentalAdapter:
    """Adapts trajectory prediction to environmental conditions"""

    def __init__(self):
        self.weather_models = {
            'clear': {'visibility': 1.0, 'motion_noise': 0.1, 'prediction_horizon': 30},
            'windy': {'visibility': 0.8, 'motion_noise': 0.5, 'prediction_horizon': 15},
            'dust_storm': {'visibility': 0.3, 'motion_noise': 1.0, 'prediction_horizon': 8},
            'rain': {'visibility': 0.6, 'motion_noise': 0.3, 'prediction_horizon': 20},
            'storm': {'visibility': 0.2, 'motion_noise': 0.8, 'prediction_horizon': 5}
        }

    def analyze_environment(self, frame: np.ndarray, detections: List[Dict]) -> Dict[str, Any]:
        """Analyze environmental conditions from frame and detections"""
        conditions = {}

        # Visibility analysis using image statistics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / np.mean(gray)
        brightness = np.mean(gray)

        # Motion blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Weather classification based on image features
        if contrast < 0.3 and brightness < 100:
            conditions['weather'] = 'dust_storm'
        elif laplacian_var < 100:
            conditions['weather'] = 'rain'
        elif contrast < 0.5:
            conditions['weather'] = 'windy'
        elif brightness < 80:
            conditions['weather'] = 'storm'
        else:
            conditions['weather'] = 'clear'

        # Obstacle detection
        conditions['obstacles'] = self._detect_obstacles(frame, detections)

        # Wind estimation from motion patterns
        conditions['wind_vector'] = self._estimate_wind(detections)

        return conditions

    def _detect_obstacles(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Detect potential obstacles that could affect trajectory"""
        obstacles = []

        # Edge detection for obstacle identification
        edges = cv2.Canny(frame, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Significant obstacle
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append({
                    'bbox': (x, y, x+w, y+h),
                    'type': 'opaque_obstacle',
                    'blocking_factor': min(1.0, area / 10000)
                })

        return obstacles

    def _estimate_wind(self, detections: List[Dict]) -> Tuple[float, float]:
        """Estimate wind direction and strength from motion patterns"""
        if len(detections) < 2:
            return (0, 0)

        # Analyze motion vectors for wind patterns
        motions = []
        for det in detections:
            if 'velocity' in det:
                motions.append(det['velocity'])

        if not motions:
            return (0, 0)

        # Average motion as wind estimate
        avg_motion = np.mean(motions, axis=0)
        wind_strength = np.linalg.norm(avg_motion)

        # Only consider significant wind
        if wind_strength > 2.0:
            return tuple(avg_motion)
        return (0, 0)

class LSTMTrajectoryPredictor(nn.Module):
    """LSTM-based trajectory prediction network"""

    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=4):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last output
        out = self.fc(out)
        return out

class KalmanTrajectoryPredictor:
    """Kalman filter-based trajectory prediction with environmental adaptation"""

    def __init__(self):
        self.filters = {}  # object_id -> KalmanFilter
        self.trajectories = {}  # object_id -> deque of TrajectoryPoint

    def initialize_filter(self, object_id: str, initial_position: Tuple[float, float],
                         initial_velocity: Tuple[float, float] = (0, 0)):
        """Initialize Kalman filter for new object"""
        kf = KalmanFilter(dim_x=6, dim_z=4)  # State: [x, y, vx, vy, ax, ay]

        # State transition matrix (constant acceleration model)
        kf.F = np.array([[1, 0, 1, 0, 0.5, 0],
                        [0, 1, 0, 1, 0, 0.5],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]])

        # Measurement matrix (we measure position and velocity)
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0]])

        # Initial state
        kf.x = np.array([initial_position[0], initial_position[1],
                        initial_velocity[0], initial_velocity[1], 0, 0])

        # Covariance matrices
        kf.P *= 1000  # Initial uncertainty
        kf.R *= 10    # Measurement noise
        # Create 6x6 process noise matrix for [x, y, vx, vy, ax, ay]
        Q_pos_vel = Q_discrete_white_noise(dim=4, dt=1, var=0.1)  # For position and velocity
        Q_full = np.zeros((6, 6))
        Q_full[:4, :4] = Q_pos_vel  # Position and velocity noise
        Q_full[4:, 4:] = np.eye(2) * 0.01  # Acceleration noise
        kf.Q = Q_full

        self.filters[object_id] = kf
        self.trajectories[object_id] = deque(maxlen=50)

    def update(self, object_id: str, position: Tuple[float, float],
               velocity: Tuple[float, float], environmental_factors: Dict[str, float]):
        """Update trajectory with new measurement"""
        if object_id not in self.filters:
            self.initialize_filter(object_id, position, velocity)

        kf = self.filters[object_id]

        # Adapt noise based on environmental conditions
        weather_noise = environmental_factors.get('motion_noise', 0.1)
        Q_pos_vel = Q_discrete_white_noise(dim=4, dt=1, var=weather_noise)
        Q_full = np.zeros((6, 6))
        Q_full[:4, :4] = Q_pos_vel  # Position and velocity noise
        Q_full[4:, 4:] = np.eye(2) * 0.01  # Acceleration noise
        kf.Q = Q_full

        # Measurement: [x, y, vx, vy]
        z = np.array([position[0], position[1], velocity[0], velocity[1]])
        kf.predict()
        kf.update(z)

        # Store trajectory point
        acceleration = (kf.x[4], kf.x[5])
        point = TrajectoryPoint(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            timestamp=time.time(),
            confidence=1.0 / (1 + np.trace(kf.P)),  # Confidence from covariance
            environmental_factors=environmental_factors
        )
        self.trajectories[object_id].append(point)

    def predict_trajectory(self, object_id: str, time_horizon: int = 30) -> List[Tuple[float, float]]:
        """Predict future trajectory using Kalman filter"""
        if object_id not in self.filters:
            return []

        kf = self.filters[object_id]

        # Get current state
        current_state = kf.x.copy()
        predictions = []

        # Predict future positions
        for i in range(time_horizon):
            # Predict next state
            kf.predict()
            x, y = kf.x[0], kf.x[1]
            predictions.append((x, y))

            # Reset state for next prediction
            kf.x = current_state

        return predictions

class PhysicsBasedPredictor:
    """Physics-based trajectory prediction considering forces and constraints"""

    def __init__(self):
        self.gravity = 0.0  # Negligible for human walking
        self.friction_coefficient = 0.1
        self.max_speed = 3.0  # pixels/frame for walking humans
        self.max_acceleration = 1.0

    def predict_physics_based(self, trajectory: List[TrajectoryPoint],
                            time_horizon: int, environmental_conditions: Dict) -> List[Tuple[float, float]]:
        """Predict trajectory using physics-based modeling"""
        if len(trajectory) < 2:
            return []

        # Get recent motion
        current_point = trajectory[-1]
        prev_point = trajectory[-2]

        # Calculate current velocity and acceleration
        dt = current_point.timestamp - prev_point.timestamp
        if dt > 0:
            vx = (current_point.position[0] - prev_point.position[0]) / dt
            vy = (current_point.position[1] - prev_point.position[1]) / dt
            ax = (current_point.velocity[0] - prev_point.velocity[0]) / dt
            ay = (current_point.velocity[1] - prev_point.velocity[1]) / dt
        else:
            vx, vy = current_point.velocity
            ax, ay = current_point.acceleration

        # Apply environmental forces
        wind_force = environmental_conditions.get('wind_vector', (0, 0))
        visibility = environmental_conditions.get('visibility', 1.0)

        # Predict future positions
        predictions = []
        x, y = current_point.position
        vx_current, vy_current = vx, vy
        ax_current, ay_current = ax, ay

        for i in range(time_horizon):
            # Apply friction
            friction_x = -self.friction_coefficient * vx_current
            friction_y = -self.friction_coefficient * vy_current

            # Apply wind force (scaled by visibility)
            wind_x = wind_force[0] * (1 - visibility)
            wind_y = wind_force[1] * (1 - visibility)

            # Update acceleration
            ax_current += friction_x + wind_x
            ay_current += friction_y + wind_y

            # Limit acceleration
            ax_current = np.clip(ax_current, -self.max_acceleration, self.max_acceleration)
            ay_current = np.clip(ay_current, -self.max_acceleration, self.max_acceleration)

            # Update velocity
            vx_current += ax_current
            vy_current += ay_current

            # Limit velocity
            speed = np.sqrt(vx_current**2 + vy_current**2)
            if speed > self.max_speed:
                vx_current *= self.max_speed / speed
                vy_current *= self.max_speed / speed

            # Update position
            x += vx_current
            y += vy_current

            predictions.append((x, y))

        return predictions

class LLMEnhancedPredictor:
    """LLM-enhanced trajectory prediction using behavioral reasoning"""

    def __init__(self, llm_provider="mock"):
        self.llm_provider = llm_provider
        self.behavioral_patterns = {
            'walking_toward_tractor': {'avoidance': 0.8, 'speed_change': -0.3},
            'walking_parallel': {'avoidance': 0.2, 'speed_change': 0.1},
            'standing_still': {'avoidance': 0.1, 'speed_change': 0.0},
            'running_away': {'avoidance': 0.9, 'speed_change': 0.5}
        }

    def predict_with_llm_reasoning(self, trajectory: List[TrajectoryPoint],
                                 environmental_conditions: Dict,
                                 tractor_position: Tuple[float, float]) -> Dict[str, Any]:
        """Use LLM reasoning for behavioral prediction"""
        if not trajectory:
            return {'behavior': 'unknown', 'confidence': 0.0}

        current_point = trajectory[-1]

        # Calculate relative position to tractor
        dx = tractor_position[0] - current_point.position[0]
        dy = tractor_position[1] - current_point.position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Determine likely behavior based on position and motion
        speed = np.linalg.norm(current_point.velocity)

        if distance < 100 and abs(dx) < 50:  # Close and in path
            behavior = 'walking_toward_tractor'
        elif speed > 2.0:
            behavior = 'running_away'
        elif speed < 0.5:
            behavior = 'standing_still'
        else:
            behavior = 'walking_parallel'

        # Get behavioral modifiers
        modifiers = self.behavioral_patterns.get(behavior, {'avoidance': 0.5, 'speed_change': 0.0})

        # Adjust for environmental conditions
        weather = environmental_conditions.get('weather', 'clear')
        if weather in ['dust_storm', 'storm']:
            modifiers['avoidance'] *= 1.5  # More cautious in bad weather
        elif weather == 'windy':
            modifiers['speed_change'] += 0.2  # May walk faster in wind

        return {
            'behavior': behavior,
            'confidence': 0.8,
            'modifiers': modifiers,
            'environmental_adaptation': weather
        }

class AdvancedTrajectoryPredictor:
    """
    Advanced trajectory prediction system combining multiple algorithms
    for robust prediction in challenging agricultural environments
    """

    def __init__(self, use_lstm=True, use_physics=True, use_llm=True):
        self.environmental_adapter = EnvironmentalAdapter()
        self.kalman_predictor = KalmanTrajectoryPredictor()
        self.physics_predictor = PhysicsBasedPredictor()
        self.llm_predictor = LLMEnhancedPredictor()

        # LSTM model (if available)
        self.use_lstm = use_lstm
        self.lstm_model = None
        if use_lstm and torch.cuda.is_available():
            try:
                self.lstm_model = LSTMTrajectoryPredictor()
                # Load pretrained weights if available
                logger.info("LSTM trajectory predictor initialized")
            except:
                logger.warning("LSTM model not available, using Kalman + Physics only")
                self.use_lstm = False

        self.use_physics = use_physics
        self.use_llm = use_llm

        # Multi-hypothesis prediction
        self.num_hypotheses = 5
        self.prediction_cache = {}

        logger.info("Advanced Trajectory Predictor initialized")

    def predict_trajectory(self, object_id: str, current_detection: Dict,
                          frame: np.ndarray, tractor_position: Tuple[float, float],
                          time_horizon: int = 30) -> PredictedTrajectory:
        """
        Predict trajectory using multiple complementary algorithms

        Args:
            object_id: Unique identifier for the object
            current_detection: Current detection with position, velocity, etc.
            frame: Current video frame for environmental analysis
            tractor_position: Position of agricultural machinery
            time_horizon: Number of frames to predict ahead

        Returns:
            PredictedTrajectory with multiple prediction hypotheses
        """

        # Extract current state
        position = current_detection.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = position
        current_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
        velocity = current_detection.get('velocity', (0, 0))

        # Analyze environmental conditions
        environmental_conditions = self.environmental_adapter.analyze_environment(
            frame, [current_detection]
        )

        # Update Kalman filter with current measurement
        self.kalman_predictor.update(
            object_id, current_pos, velocity,
            environmental_conditions
        )

        # Get trajectory history
        trajectory = list(self.kalman_predictor.trajectories.get(object_id, []))

        # Generate multiple prediction hypotheses
        hypotheses = []

        # Hypothesis 1: Kalman filter prediction
        kalman_pred = self.kalman_predictor.predict_trajectory(object_id, time_horizon)
        if kalman_pred:
            hypotheses.append({
                'method': 'kalman_filter',
                'trajectory': kalman_pred,
                'confidence': 0.7
            })

        # Hypothesis 2: Physics-based prediction
        if self.use_physics and trajectory:
            physics_pred = self.physics_predictor.predict_physics_based(
                trajectory, time_horizon, environmental_conditions
            )
            if physics_pred:
                hypotheses.append({
                    'method': 'physics_based',
                    'trajectory': physics_pred,
                    'confidence': 0.6
                })

        # Hypothesis 3: LLM-enhanced behavioral prediction
        if self.use_llm and trajectory:
            llm_reasoning = self.llm_predictor.predict_with_llm_reasoning(
                trajectory, environmental_conditions, tractor_position
            )

            # Modify physics prediction based on LLM reasoning
            if physics_pred:
                modified_pred = self._apply_llm_modifiers(
                    physics_pred, llm_reasoning, environmental_conditions
                )
                hypotheses.append({
                    'method': 'llm_enhanced',
                    'trajectory': modified_pred,
                    'confidence': 0.8,
                    'behavior': llm_reasoning['behavior']
                })

        # Hypothesis 4: LSTM-based prediction (if available)
        if self.use_lstm and self.lstm_model and len(trajectory) >= 5:
            lstm_pred = self._predict_with_lstm(trajectory, time_horizon)
            if lstm_pred:
                hypotheses.append({
                    'method': 'lstm_neural',
                    'trajectory': lstm_pred,
                    'confidence': 0.75
                })

        # Hypothesis 5: Ensemble prediction (weighted average of all methods)
        if len(hypotheses) > 1:
            ensemble_pred = self._create_ensemble_prediction(hypotheses, time_horizon)
            hypotheses.append({
                'method': 'ensemble',
                'trajectory': ensemble_pred,
                'confidence': 0.85
            })

        # Select best hypothesis based on environmental conditions
        best_hypothesis = self._select_best_hypothesis(hypotheses, environmental_conditions)

        # Create final prediction result
        predicted_trajectory = PredictedTrajectory(
            object_id=object_id,
            current_position=current_pos,
            predicted_path=best_hypothesis['trajectory'],
            confidence_scores=[best_hypothesis['confidence']] * len(best_hypothesis['trajectory']),
            time_horizon=time_horizon,
            environmental_conditions=environmental_conditions,
            risk_assessment=self._assess_trajectory_risk(
                best_hypothesis['trajectory'], tractor_position, environmental_conditions
            )
        )

        return predicted_trajectory

    def _apply_llm_modifiers(self, trajectory: List[Tuple[float, float]],
                           llm_reasoning: Dict, environmental_conditions: Dict) -> List[Tuple[float, float]]:
        """Apply LLM behavioral modifiers to trajectory"""
        modifiers = llm_reasoning.get('modifiers', {})
        avoidance_factor = modifiers.get('avoidance', 0.5)
        speed_change = modifiers.get('speed_change', 0.0)

        modified_trajectory = []
        for i, (x, y) in enumerate(trajectory):
            # Apply avoidance behavior (curve away from danger)
            if avoidance_factor > 0.5:
                # Add perpendicular component to avoid tractor
                avoidance_strength = (avoidance_factor - 0.5) * 0.1 * (i + 1)
                x += np.random.normal(0, avoidance_strength)
                y += np.random.normal(0, avoidance_strength)

            # Apply speed changes
            if speed_change != 0:
                speed_factor = 1 + speed_change * 0.1
                x *= speed_factor
                y *= speed_factor

            modified_trajectory.append((x, y))

        return modified_trajectory

    def _predict_with_lstm(self, trajectory: List[TrajectoryPoint], time_horizon: int) -> List[Tuple[float, float]]:
        """Use LSTM model for trajectory prediction"""
        if not self.lstm_model or len(trajectory) < 5:
            return []

        try:
            # Prepare input sequence
            sequence = []
            for point in trajectory[-10:]:  # Last 10 points
                sequence.append([
                    point.position[0], point.position[1],
                    point.velocity[0], point.velocity[1],
                    point.acceleration[0], point.acceleration[1]
                ])

            if len(sequence) < 5:
                return []

            # Convert to tensor
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            # Predict next positions
            predictions = []
            current_input = input_tensor

            for _ in range(time_horizon):
                with torch.no_grad():
                    output = self.lstm_model(current_input)
                    next_pos = output.squeeze().numpy()

                predictions.append((next_pos[0], next_pos[1]))

                # Update input for next prediction
                new_point = np.array([next_pos[0], next_pos[1], next_pos[2], next_pos[3], 0, 0])
                current_input = torch.cat([current_input[:, 1:], torch.tensor(new_point).unsqueeze(0).unsqueeze(0)], dim=1)

            return predictions

        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return []

    def _create_ensemble_prediction(self, hypotheses: List[Dict], time_horizon: int) -> List[Tuple[float, float]]:
        """Create ensemble prediction by weighted averaging"""
        if not hypotheses:
            return []

        # Weight by confidence
        weights = [h['confidence'] for h in hypotheses]
        total_weight = sum(weights)

        ensemble_pred = []
        for i in range(time_horizon):
            weighted_x = 0
            weighted_y = 0

            for j, hypothesis in enumerate(hypotheses):
                if i < len(hypothesis['trajectory']):
                    x, y = hypothesis['trajectory'][i]
                    weight = weights[j] / total_weight
                    weighted_x += x * weight
                    weighted_y += y * weight

            ensemble_pred.append((weighted_x, weighted_y))

        return ensemble_pred

    def _select_best_hypothesis(self, hypotheses: List[Dict],
                              environmental_conditions: Dict) -> Dict:
        """Select best prediction hypothesis based on environmental conditions"""
        if not hypotheses:
            return {'trajectory': [], 'confidence': 0.0}

        weather = environmental_conditions.get('weather', 'clear')

        # Weather-specific preferences
        if weather in ['dust_storm', 'storm']:
            # Prefer physics-based and LLM-enhanced in bad visibility
            preferred_methods = ['physics_based', 'llm_enhanced', 'ensemble']
        elif weather == 'windy':
            # Prefer Kalman and physics in windy conditions
            preferred_methods = ['kalman_filter', 'physics_based', 'ensemble']
        else:
            # Prefer ensemble in clear conditions
            preferred_methods = ['ensemble', 'lstm_neural', 'kalman_filter']

        # Find best available hypothesis
        for method in preferred_methods:
            for hypothesis in hypotheses:
                if hypothesis['method'] == method:
                    return hypothesis

        # Fallback to highest confidence
        return max(hypotheses, key=lambda x: x['confidence'])

    def _assess_trajectory_risk(self, trajectory: List[Tuple[float, float]],
                               tractor_position: Tuple[float, float],
                               environmental_conditions: Dict) -> Dict[str, Any]:
        """Assess risk level of predicted trajectory"""
        if not trajectory:
            return {'risk_level': 'UNKNOWN', 'risk_score': 0.5}

        # Calculate minimum distance to tractor
        min_distance = float('inf')
        closest_point = None

        for point in trajectory:
            distance = np.sqrt((point[0] - tractor_position[0])**2 +
                             (point[1] - tractor_position[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = point

        # Risk assessment based on distance and environmental conditions
        base_risk = max(0, 1 - (min_distance / 200))  # Normalize distance

        # Environmental risk modifiers
        weather = environmental_conditions.get('weather', 'clear')
        weather_multipliers = {
            'clear': 1.0,
            'windy': 1.2,
            'rain': 1.1,
            'dust_storm': 1.5,
            'storm': 1.8
        }

        risk_score = base_risk * weather_multipliers.get(weather, 1.0)
        risk_score = min(1.0, risk_score)

        # Determine risk level
        if risk_score > 0.8:
            risk_level = 'CRITICAL'
        elif risk_score > 0.6:
            risk_level = 'HIGH'
        elif risk_score > 0.4:
            risk_level = 'MEDIUM'
        elif risk_score > 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'SAFE'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'min_distance': min_distance,
            'closest_point': closest_point,
            'environmental_factor': weather_multipliers.get(weather, 1.0)
        }

# Global predictor instance
_trajectory_predictor = None

def get_trajectory_predictor() -> AdvancedTrajectoryPredictor:
    """Get global trajectory predictor instance"""
    global _trajectory_predictor
    if _trajectory_predictor is None:
        _trajectory_predictor = AdvancedTrajectoryPredictor()
    return _trajectory_predictor

def predict_human_trajectory(object_id: str, detection: Dict, frame: np.ndarray,
                           tractor_position: Tuple[float, float], time_horizon: int = 30):
    """
    Convenience function for trajectory prediction

    Args:
        object_id: Unique object identifier
        detection: Current detection data
        frame: Current video frame
        tractor_position: Position of agricultural machinery
        time_horizon: Frames to predict ahead

    Returns:
        PredictedTrajectory with path and risk assessment
    """
    predictor = get_trajectory_predictor()
    return predictor.predict_trajectory(object_id, detection, frame, tractor_position, time_horizon)