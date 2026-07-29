import cv2
import numpy as np
import logging

class HomographyTransformer:
    """
    Transform image coordinates to a ground plane using calibrated homography.
    """
    def __init__(self, src_points=None, dst_scale=(20.0, 40.0)):
        """
        Initialize homography transformer.

        Args:
            src_points (list): Four image points in pixel coordinates.
            dst_scale (tuple): Width and depth of destination ground plane in meters.
        """
        self.src_points = src_points or [(120, 460), (520, 460), (540, 300), (100, 300)]
        self.dst_scale = dst_scale
        self.dst_points = np.array([
            [0.0, 0.0],
            [dst_scale[0], 0.0],
            [dst_scale[0], dst_scale[1]],
            [0.0, dst_scale[1]],
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(np.array(self.src_points, dtype=np.float32), self.dst_points)
        if self.H is None:
            raise ValueError("Unable to compute homography with given source points")
        logging.info("Homography transformer initialized")

    def transform_point(self, point):
        """
        Transform an image point to ground-plane coordinates.

        Args:
            point (tuple): Image point (x, y).

        Returns:
            tuple: Ground-plane coordinates in meters (x, z).
        """
        p = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(p, self.H)
        return tuple(transformed[0][0])

    def project_bbox_to_ground_plane(self, bbox):
        """
        Project the bottom-center of a bounding box to the ground plane.

        Args:
            bbox (tuple): Bounding box (x1, y1, x2, y2).

        Returns:
            tuple: Ground-plane point (x_meters, z_meters).
        """
        x1, y1, x2, y2 = bbox
        bottom_center = ((x1 + x2) / 2.0, y2)
        return self.transform_point(bottom_center)

    def set_points(self, points):
        """
        Set new source points.

        Args:
            points (list): 4 points.
        """
        self.src_points = points
        self.H, _ = cv2.findHomography(np.array(self.src_points, dtype=np.float32), self.dst_points)
