import cv2
import numpy as np
import logging

class VideoStabilizer:
    """
    Video stabilization to reduce vibration and improve detection/tracking robustness.
    """
    def __init__(self, max_history=20, feature_count=200):
        self.prev_gray = None
        self.transforms = []
        self.max_history = max_history
        self.feature_count = feature_count
        logging.info("Video stabilizer initialized")

    def stabilize(self, frame):
        """
        Stabilize the current video frame and smooth camera motion.

        Args:
            frame (numpy array): Current BGR frame.

        Returns:
            numpy array: Stabilized frame.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=self.feature_count, qualityLevel=0.01, minDistance=8, blockSize=7)
        if prev_pts is None:
            self.prev_gray = curr_gray
            return frame

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None:
            self.prev_gray = curr_gray
            return frame

        valid = status.flatten() == 1
        if np.count_nonzero(valid) < 10:
            self.prev_gray = curr_gray
            return frame

        prev_pts = prev_pts[valid]
        curr_pts = curr_pts[valid]

        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
        if transform is None:
            self.prev_gray = curr_gray
            return frame

        transform_matrix = np.vstack([transform, [0, 0, 1]]).astype(np.float32)
        self.transforms.append(transform_matrix)
        if len(self.transforms) > self.max_history:
            self.transforms.pop(0)

        smoothed = np.mean(self.transforms, axis=0)
        stabilized = cv2.warpPerspective(frame, smoothed, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

        self.prev_gray = curr_gray
        return stabilized

class OcclusionHead:
    """
    Estimate occlusion confidence from tracking history and detection overlap.
    """
    def __init__(self, iou_threshold=0.35, min_track_length=3):
        self.iou_threshold = iou_threshold
        self.min_track_length = min_track_length
        logging.info("Occlusion head initialized")

    @staticmethod
    def bbox_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0

        boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0.0

    def predict_occlusion(self, bbox, detections, is_predicted=False, track_history=None):
        """
        Estimate whether the current object is occluded.

        Args:
            bbox (tuple): Current track bounding box.
            detections (list): Recent detection list [(bbox, conf), ...].
            is_predicted (bool): True if the tracker is predicting due to missed detection.
            track_history (list): History of previous bboxes.

        Returns:
            dict: Occlusion data.
        """
        result = {
            'is_occluded': False,
            'occlusion_confidence': 0.0,
            'reason': 'visible',
        }

        if is_predicted:
            result['is_occluded'] = True
            result['occlusion_confidence'] = 0.85
            result['reason'] = 'predicted by tracker'
            return result

        if not detections:
            return result

        overlaps = [self.bbox_iou(bbox, det_bbox) for det_bbox, _ in detections]
        max_overlap = max(overlaps) if overlaps else 0.0
        if max_overlap < self.iou_threshold:
            result['is_occluded'] = True
            result['occlusion_confidence'] = 0.75
            result['reason'] = 'low overlap with detections'
        elif track_history is not None and len(track_history) >= self.min_track_length:
            result['is_occluded'] = False
            result['occlusion_confidence'] = 0.2
            result['reason'] = 'stable visible track'

        return result