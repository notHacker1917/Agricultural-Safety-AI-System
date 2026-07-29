"""
ByteTrack Implementation for Agricultural Safety AI System
Replaces DeepSORT with ByteTrack for better occlusion handling and identity consistency.

ByteTrack associates every detection (high and low confidence) instead of only using
high-confidence detections, which significantly improves tracking in crowded scenes
and during occlusions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class STrack:
    """Single object track data structure."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    state: str = 'new'  # new, tracked, lost, removed
    is_activated: bool = False
    tracklet_len: int = 0
    start_frame: int = 0
    frame_id: int = 0
    last_predict_frame: int = 0
    
    # Kalman filter state
    mean: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float) * 10)
    
    # Velocity estimation
    velocity: Tuple[float, float] = (0.0, 0.0)
    velocity_history: List[Tuple[float, float]] = field(default_factory=list)
    
    # Occlusion handling
    occlusion_frames: int = 0
    max_occlusion_frames: int = 30
    is_occluded: bool = False
    
    # Trajectory history
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    max_trajectory_length: int = 100
    
    # Confidence decay during occlusion
    confidence_decay: float = 0.95
    
    def __post_init__(self):
        # Initialize mean with bbox
        self.mean[:4, 0] = self.bbox
        self.mean = self.mean.astype(float)
        self.covariance = self.covariance.astype(float)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary for compatibility."""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'center': self.center,
            'velocity': self.velocity,
            'is_occluded': self.is_occluded,
            'occlusion_frames': self.occlusion_frames,
            'state': self.state,
            'trajectory': self.trajectory[-30:],  # Last 30 positions
            'tracklet_len': self.tracklet_len
        }
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float, frame_id: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.last_predict_frame = frame_id
        self.is_occluded = False
        self.occlusion_frames = 0
        
        # Update trajectory
        self.trajectory.append(self.center)
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory = self.trajectory[-self.max_trajectory_length:]
        
        # Update velocity
        if len(self.trajectory) >= 2:
            prev_center = self.trajectory[-2]
            curr_center = self.trajectory[-1]
            vel = (curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
            
            # Smooth velocity
            self.velocity_history.append(vel)
            if len(self.velocity_history) > 5:
                self.velocity_history = self.velocity_history[-5:]
            
            avg_vel = (
                np.mean([v[0] for v in self.velocity_history]),
                np.mean([v[1] for v in self.velocity_history])
            )
            self.velocity = (avg_vel[0], avg_vel[1])
        
        if self.state == 'new':
            self.is_activated = True
            self.state = 'tracked'
    
    def predict(self, frame_id: int) -> Tuple[float, float, float, float]:
        """Predict next position based on velocity."""
        if not self.is_occluded:
            # Simple linear prediction based on velocity
            predicted_bbox = (
                self.bbox[0] + self.velocity[0],
                self.bbox[1] + self.velocity[1],
                self.bbox[2] + self.velocity[0],
                self.bbox[3] + self.velocity[1]
            )
            self.mean[:4, 0] = predicted_bbox
        else:
            # During occlusion, apply confidence decay
            self.confidence *= self.confidence_decay
            self.occlusion_frames += 1
        
        self.last_predict_frame = frame_id
        return self.mean[:4, 0].tolist()


class ByteTrack:
    """
    ByteTrack multi-object tracker.
    
    Key features:
    - Associates ALL detections (high and low confidence)
    - Two-stage matching: high confidence first, then low confidence
    - Better handling of occlusions and crowded scenes
    - Simple and efficient IoU-based matching
    """
    
    def __init__(self, config=None):
        """
        Initialize ByteTrack.
        
        Args:
            config: Configuration object with tracking parameters
        """
        # Default configuration
        self.track_threshold = 0.5        # High confidence threshold
        self.low_match_threshold = 0.1    # Low confidence threshold
        self.new_track_threshold = 0.6    # Threshold for new track confirmation
        self.track_buffer = 30            # Frames to keep lost tracks
        self.match_threshold = 0.8        # IoU matching threshold
        self.min_box_area = 100.0         # Minimum bbox area
        self.max_occlusion_frames = 30    # Max frames for occlusion recovery
        
        # Override with config if provided
        if config is not None:
            self._apply_config(config)
        
        self.frame_id = 0
        self.next_track_id = 0
        self.tracked_stracks = []    # Successfully tracked tracks
        self.lost_stracks = []       # Temporarily lost tracks
        self.removed_stracks = []    # Permanently removed tracks
        
        # For debugging
        self.total_tracks_created = 0
        self.tracks_recovered = 0
        
    def _apply_config(self, config):
        """Apply configuration parameters."""
        if hasattr(config, 'tracking'):
            cfg = config.tracking
            self.track_threshold = cfg.track_threshold
            self.low_match_threshold = cfg.low_match_threshold
            self.new_track_threshold = cfg.new_track_threshold
            self.track_buffer = cfg.track_buffer
            self.match_threshold = cfg.match_threshold
            self.min_box_area = cfg.min_box_area
            self.max_occlusion_frames = cfg.max_occlusion_frames
    
    def update(self, detections: List[Dict], frame_id: Optional[int] = None) -> List[STrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            frame_id: Optional frame ID
            
        Returns:
            List of active STrack objects
        """
        if frame_id is not None:
            self.frame_id = frame_id
        else:
            self.frame_id += 1
        
        # Extract detection bboxes and confidences
        dets = [d['bbox'] for d in detections]
        confs = [d.get('confidence', 0.5) for d in detections]
        
        # Separate high and low confidence detections
        high_conf_indices = []
        low_conf_indices = []
        
        for i, conf in enumerate(confs):
            # Filter by minimum area
            x1, y1, x2, y2 = dets[i]
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue
            
            if conf >= self.track_threshold:
                high_conf_indices.append(i)
            elif conf >= self.low_match_threshold:
                low_conf_indices.append(i)
        
        # Create STrack objects for detections
        detected_stracks = []
        for i in high_conf_indices + low_conf_indices:
            is_high_conf = i in high_conf_indices
            strack = STrack(
                track_id=-1,  # Will be assigned later
                bbox=dets[i],
                confidence=confs[i],
                state='new'
            )
            strack.frame_id = self.frame_id
            strack.start_frame = self.frame_id
            strack.last_predict_frame = self.frame_id
            detected_stracks.append(strack)
        
        # Step 1: Match high confidence detections with tracked tracks
        high_conf_dets = [s for s in detected_stracks if confs[detected_stracks.index(s)] >= self.track_threshold]
        low_conf_dets = [s for s in detected_stracks if confs[detected_stracks.index(s)] < self.track_threshold]
        
        # Get currently tracked tracks
        active_tracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        
        # First association: match tracked tracks with high confidence detections
        matches, u_tracks, u_dets = self._match(active_tracks, high_conf_dets)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            det = high_conf_dets[det_idx]
            track.update(det.bbox, det.confidence, self.frame_id)
        
        # Second association: match remaining tracks with low confidence detections
        remaining_tracks = [active_tracks[i] for i in u_tracks]
        matches2, u_tracks2, u_dets2 = self._match(remaining_tracks, low_conf_dets)
        
        # Update matched tracks from low confidence
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            det = low_conf_dets[det_idx]
            track.update(det.bbox, det.confidence, self.frame_id)
        
        # Handle unmatched tracks (potentially occluded)
        for track_idx in u_tracks2:
            track = remaining_tracks[track_idx]
            track.is_occluded = True
            track.occlusion_frames += 1
            
            if track.occlusion_frames < self.max_occlusion_frames:
                # Keep in lost tracks for potential recovery
                track.state = 'lost'
                self.lost_stracks.append(track)
            else:
                # Remove track after max occlusion frames
                track.state = 'removed'
                self.removed_stracks.append(track)
        
        # Handle unmatched detections (create new tracks)
        unmatched_dets = [low_conf_dets[i] for i in u_dets2] + [high_conf_dets[i] for i in u_dets]
        
        # Try to recover from lost tracks first
        if unmatched_dets and self.lost_stracks:
            lost_tracks = [t for t in self.lost_stracks if t.occlusion_frames < self.max_occlusion_frames]
            recovery_matches, u_lost, u_unmatched = self._match(lost_tracks, unmatched_dets, threshold=0.5)
            
            for lost_idx, det_idx in recovery_matches:
                track = lost_tracks[lost_idx]
                det = unmatched_dets[det_idx]
                track.update(det.bbox, det.confidence, self.frame_id)
                track.state = 'tracked'
                self.tracks_recovered += 1
            
            # Remove recovered tracks from lost list
            self.lost_stracks = [t for i, t in enumerate(self.lost_stracks) if i not in u_lost]
            unmatched_dets = [unmatched_dets[i] for i in u_unmatched]
        
        # Create new tracks for remaining unmatched detections
        for det in unmatched_dets:
            if det.confidence >= self.new_track_threshold:
                new_track = STrack(
                    track_id=self.next_track_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    state='new'
                )
                new_track.frame_id = self.frame_id
                new_track.start_frame = self.frame_id
                new_track.last_predict_frame = self.frame_id
                self.next_track_id += 1
                self.total_tracks_created += 1
                self.tracked_stracks.append(new_track)
        
        # Update tracked tracks list
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state in ['tracked', 'new']]
        
        # Clean up old lost tracks
        self.lost_stracks = [t for t in self.lost_stracks 
                           if self.frame_id - t.last_predict_frame < self.track_buffer]
        
        return self.tracked_stracks
    
    def _match(self, tracks: List[STrack], detections: List[STrack], 
               threshold: float = None) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks with detections using IoU.
        
        Returns:
            matches: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_dets: List of unmatched detection indices
        """
        if threshold is None:
            threshold = self.match_threshold
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        # Hungarian algorithm for optimal matching
        import scipy.optimize
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= (1.0 - threshold):
                matches.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_dets.remove(j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_active_tracks(self) -> List[Dict]:
        """Get all active tracks as dictionaries."""
        return [track.to_dict() for track in self.tracked_stracks 
                if track.state in ['tracked', 'new']]
    
    def get_track_by_id(self, track_id: int) -> Optional[STrack]:
        """Get a specific track by ID."""
        for track in self.tracked_stracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get trajectory history for a track."""
        track = self.get_track_by_id(track_id)
        if track:
            return track.trajectory
        return []
    
    def reset(self):
        """Reset tracker state."""
        self.frame_id = 0
        self.next_track_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.total_tracks_created = 0
        self.tracks_recovered = 0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            'total_tracks_created': self.total_tracks_created,
            'tracks_recovered': self.tracks_recovered,
            'active_tracks': len([t for t in self.tracked_stracks if t.state == 'tracked']),
            'lost_tracks': len(self.lost_stracks),
            'frame_id': self.frame_id
        }


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack that provides compatibility with existing pipeline.
    Converts between detection format and track format.
    """
    
    def __init__(self, config=None):
        """
        Initialize ByteTrack wrapper.
        
        Args:
            config: Configuration object
        """
        self.tracker = ByteTrack(config)
        self.config = config
        logger.info("ByteTrack wrapper initialized")
    
    def update(self, detections: List[Dict], frame_id: Optional[int] = None) -> List[Dict]:
        """
        Update tracker and return tracked objects.
        
        Args:
            detections: List of detection dictionaries
            frame_id: Optional frame ID
            
        Returns:
            List of tracked object dictionaries
        """
        # Run ByteTrack
        tracks = self.tracker.update(detections, frame_id)
        
        # Convert to output format
        results = []
        for track in tracks:
            if track.state in ['tracked', 'new']:
                result = {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'center': track.center,
                    'velocity': track.velocity,
                    'is_occluded': track.is_occluded,
                    'occlusion_frames': track.occlusion_frames,
                    'trajectory': track.trajectory[-30:],
                    'tracklet_len': track.tracklet_len
                }
                results.append(result)
        
        return results
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Get trajectory for a track."""
        return self.tracker.get_trajectory(track_id)
    
    def get_active_tracks(self) -> List[Dict]:
        """Get all active tracks."""
        return self.tracker.get_active_tracks()
    
    def reset(self):
        """Reset tracker."""
        self.tracker.reset()
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return self.tracker.get_stats()


def test_bytetrack():
    """Test ByteTrack implementation."""
    logger.info("Testing ByteTrack implementation")
    
    # Create tracker
    tracker = ByteTrack()
    
    # Simulate detections
    detections = [
        {'bbox': (100, 100, 150, 200), 'confidence': 0.9},
        {'bbox': (300, 150, 350, 250), 'confidence': 0.8},
        {'bbox': (500, 200, 550, 300), 'confidence': 0.7},
    ]
    
    # Update tracker
    tracks = tracker.update(detections, frame_id=1)
    logger.info(f"Frame 1: {len(tracks)} tracks")
    
    # Simulate second frame with slightly moved objects
    detections2 = [
        {'bbox': (105, 105, 155, 205), 'confidence': 0.85},
        {'bbox': (305, 155, 355, 255), 'confidence': 0.75},
        {'bbox': (505, 205, 555, 305), 'confidence': 0.65},
    ]
    
    tracks2 = tracker.update(detections2, frame_id=2)
    logger.info(f"Frame 2: {len(tracks2)} tracks")
    
    # Get stats
    stats = tracker.get_stats()
    logger.info(f"Tracker stats: {stats}")
    
    logger.info("ByteTrack test complete")


if __name__ == '__main__':
    test_bytetrack()