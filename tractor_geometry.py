"""
Tractor Geometry & Field of View Analysis

Models realistic tractor dimensions, camera mounting, and field-of-view geometry
to compute actual distance measurements from pixel coordinates.

REALISM CHECK:
- Average combine harvester: 3m wide × 3m tall × 10m long
- Camera mounting height: ~2.5m (operator cab level)
- Horizontal FOV: 90-110° (wide angle for safety)
- Vertical FOV: 60-75° (covers ground ahead)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TractorModel(Enum):
    """Common harvester models with specifications."""
    CLAAS_LEXION = "CLAAS_LEXION"  # 3.5m width
    JOHN_DEERE_S780 = "JOHN_DEERE_S780"  # 3.81m width
    MASSEY_FERGUSON = "MASSEY_FERGUSON"  # 3.2m width
    FENDT_IDEAL = "FENDT_IDEAL"  # 3.7m width
    GENERIC = "GENERIC"  # Default 3m width


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for distance computation."""
    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int = 1920
    height: int = 1080
    
    def __post_init__(self):
        """Compute derived metrics."""
        self.horizontal_fov = 2 * np.arctan(self.width / (2 * self.fx))
        self.vertical_fov = 2 * np.arctan(self.height / (2 * self.fy))
        logger.info(f"Camera H-FOV: {np.degrees(self.horizontal_fov):.1f}° V-FOV: {np.degrees(self.vertical_fov):.1f}°")


@dataclass
class TractorGeometry:
    """Physical tractor dimensions and sensor placement."""
    model: TractorModel
    width: float  # Vehicle width in meters
    height: float  # Vehicle height in meters
    length: float  # Vehicle length in meters
    camera_height: float  # Camera mounting height above ground (m)
    camera_forward_offset: float  # Distance forward from front axle (m)
    camera_lateral_offset: float  # Lateral offset from centerline (m)
    camera_pitch: float  # Camera pitch angle in radians (negative = looking down)
    camera_roll: float  # Camera roll angle in radians
    camera_yaw: float  # Camera yaw angle in radians (0 = forward)
    safety_radius_front: float  # Danger zone radius in front (m)
    safety_radius_side: float  # Danger zone radius to sides (m)
    
    @classmethod
    def default_harvester(cls, model: TractorModel = TractorModel.GENERIC) -> "TractorGeometry":
        """Standard harvester configuration."""
        configs = {
            TractorModel.GENERIC: {
                "width": 3.0,
                "height": 3.0,
                "length": 10.0,
                "camera_height": 2.5,
                "camera_forward_offset": 1.5,
                "camera_lateral_offset": 0.0,
                "camera_pitch": -np.radians(15),  # 15° down
                "camera_roll": 0.0,
                "camera_yaw": 0.0,
                "safety_radius_front": 1.5,
                "safety_radius_side": 2.0,
            },
            TractorModel.CLAAS_LEXION: {
                "width": 3.5,
                "height": 3.2,
                "length": 10.5,
                "camera_height": 2.6,
                "camera_forward_offset": 1.7,
                "camera_lateral_offset": 0.0,
                "camera_pitch": -np.radians(15),
                "camera_roll": 0.0,
                "camera_yaw": 0.0,
                "safety_radius_front": 1.8,
                "safety_radius_side": 2.2,
            },
            TractorModel.JOHN_DEERE_S780: {
                "width": 3.81,
                "height": 3.1,
                "length": 10.8,
                "camera_height": 2.55,
                "camera_forward_offset": 1.6,
                "camera_lateral_offset": 0.0,
                "camera_pitch": -np.radians(15),
                "camera_roll": 0.0,
                "camera_yaw": 0.0,
                "safety_radius_front": 1.9,
                "safety_radius_side": 2.3,
            }
        }
        
        cfg = configs.get(model, configs[TractorModel.GENERIC])
        cfg["model"] = model
        return cls(**cfg)


class TractorPOVGeometry:
    """
    Computes actual 3D positions of detected people relative to tractor
    using camera intrinsics and extrinsic geometry.
    """
    
    def __init__(
        self,
        tractor: TractorGeometry,
        camera: CameraIntrinsics,
        assume_ground_plane: bool = True
    ):
        self.tractor = tractor
        self.camera = camera
        self.assume_ground_plane = assume_ground_plane
        
        # Camera position in tractor coordinate frame
        self.camera_position = np.array([
            tractor.camera_lateral_offset,
            tractor.camera_height,
            tractor.camera_forward_offset
        ])
        
        # Rotation matrix from camera to tractor frame
        self.R_cam_to_tractor = self._build_rotation_matrix(
            tractor.camera_pitch,
            tractor.camera_roll,
            tractor.camera_yaw
        )
        
        logger.info(f"TractorPOV initialized: {tractor.model.value}, Camera @ ({self.camera_position[0]:.2f}, {self.camera_position[1]:.2f}, {self.camera_position[2]:.2f})m")
    
    @staticmethod
    def _build_rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
        """Build 3x3 rotation matrix from Euler angles (pitch, roll, yaw)."""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def pixel_to_3d_ground_plane(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float, float]]:
        """
        Project pixel coordinate to 3D tractor-relative position (X, Y, Z).
        
        Assumes person stands on ground plane (Y=0).
        Uses camera intrinsics + ground plane constraint.
        
        Args:
            pixel_x: X coordinate in image (pixels)
            pixel_y: Y coordinate in image (pixels)
            
        Returns:
            (x, y, z) in tractor frame, or None if projection fails
        """
        # Normalized image coordinates (camera looks along +Z)
        norm_x = (pixel_x - self.camera.cx) / self.camera.fx
        norm_y = (pixel_y - self.camera.cy) / self.camera.fy
        
        # Ray direction in camera frame: (x/f, y/f, 1) normalized
        ray_cam = np.array([norm_x, norm_y, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform ray to tractor frame using inverse of rotation
        # (R^T instead of R since we're transforming direction vectors)
        ray_tractor = self.R_cam_to_tractor.T @ ray_cam
        
        # Ray line: P(t) = camera_pos + t * ray_tractor
        # Ground plane constraint: Y = 0
        # Solve: camera_pos[1] + t * ray_tractor[1] = 0
        
        if abs(ray_tractor[1]) < 1e-6:
            return None  # Ray parallel to ground plane
        
        t = -self.camera_position[1] / ray_tractor[1]
        
        if t <= 0.01:
            return None  # Point behind camera
        
        intersection = self.camera_position + t * ray_tractor
        return tuple(intersection)
    
    def compute_distance_meters(self, pixel_x: float, pixel_y: float) -> Optional[float]:
        """
        Compute actual distance from camera to point in image.
        
        Returns distance in meters (Euclidean distance from camera).
        """
        pos_3d = self.pixel_to_3d_ground_plane(pixel_x, pixel_y)
        if pos_3d is None:
            return None
        
        x, y, z = pos_3d
        # Distance from camera position
        distance = np.sqrt((x - self.camera_position[0])**2 + 
                          (y - self.camera_position[1])**2 + 
                          (z - self.camera_position[2])**2)
        return distance
    
    def pixel_bbox_to_3d_position(self, bbox: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """
        Convert detection bounding box (x1, y1, x2, y2) to 3D ground position.
        
        Uses bottom-center of bbox (person's feet) as ground contact point.
        
        Returns:
            (x, y, z, distance_meters) or None if projection fails
        """
        x1, y1, x2, y2 = bbox
        # Bottom-center of bbox (where person stands)
        pixel_x = (x1 + x2) / 2
        pixel_y = y2  # Bottom of bbox
        
        pos_3d = self.pixel_to_3d_ground_plane(pixel_x, pixel_y)
        if pos_3d is None:
            return None
        
        distance = self.compute_distance_meters(pixel_x, pixel_y)
        if distance is None:
            return None
        
        x, y, z = pos_3d
        return (x, y, z, distance)
    
    def compute_safety_zone(self) -> np.ndarray:
        """
        Compute ground-plane safety zone polygon in tractor coordinates.
        
        Returns:
            Nx3 array of vertices (x, y, z) forming safety zone boundary
        """
        front_dist = self.tractor.safety_radius_front
        side_dist = self.tractor.safety_radius_side
        
        # Safety zone polygon (looking from above, Y-axis up in tractor view)
        # Front-left, Front-right, Back-right, Back-left
        zone = np.array([
            [-side_dist, 0, front_dist],         # Front-left
            [side_dist, 0, front_dist],           # Front-right
            [side_dist, 0, -1.0],                 # Back-right
            [-side_dist, 0, -1.0],                # Back-left
        ])
        return zone
    
    def is_in_safety_zone(self, x: float, z: float) -> bool:
        """
        Check if ground position (x, z) is in safety zone.
        
        Args:
            x: Lateral position (m, 0 = centerline)
            z: Longitudinal position (m, >0 = in front)
        """
        if z < -1.0 or z > self.tractor.safety_radius_front:
            return False
        
        if abs(x) > self.tractor.safety_radius_side:
            return False
        
        # Linear interpolation: zone narrows with distance
        if z >= 0:
            max_side = self.tractor.safety_radius_side * (1.0 - z / self.tractor.safety_radius_front)
        else:
            max_side = self.tractor.safety_radius_side
        
        return abs(x) <= max_side
    
    def point_to_safety_zone_distance(self, x: float, z: float) -> float:
        """
        Compute distance from point to nearest safety zone boundary.
        
        Returns:
            Distance in meters (negative = inside zone, positive = outside)
        """
        if self.is_in_safety_zone(x, z):
            # Minimum distance to any boundary
            dist_front = self.tractor.safety_radius_front - z if z >= 0 else float('inf')
            dist_side = self.tractor.safety_radius_side - abs(x)
            dist_back = z + 1.0
            return -(min(dist_front, dist_side, dist_back))
        
        # Distance to zone edge
        dist_x = max(0, abs(x) - self.tractor.safety_radius_side)
        dist_z = max(0, max(-z - 1.0, z - self.tractor.safety_radius_front))
        return np.sqrt(dist_x**2 + dist_z**2)


# ============= UTILITY FUNCTIONS =============

def create_realistic_camera() -> CameraIntrinsics:
    """Create typical agricultural camera (wide angle, ~90° FOV)."""
    # Typical wide-angle camera specs
    width, height = 1920, 1080
    fov_degrees = 95
    
    # f = (width/2) / tan(fov/2)
    f = (width / 2) / np.tan(np.radians(fov_degrees / 2))
    
    return CameraIntrinsics(
        fx=f,
        fy=f,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height
    )


def test_geometry():
    """Quick test of tractor POV geometry."""
    logger.info("=" * 70)
    logger.info("TRACTOR POV GEOMETRY TEST")
    logger.info("=" * 70)
    
    tractor = TractorGeometry.default_harvester(TractorModel.CLAAS_LEXION)
    camera = create_realistic_camera()
    pov = TractorPOVGeometry(tractor, camera)
    
    logger.info(f"\nTractor: {tractor.model.value}")
    logger.info(f"  Dimensions: {tractor.width:.1f}m W × {tractor.height:.1f}m H × {tractor.length:.1f}m L")
    logger.info(f"  Camera: {tractor.camera_height:.1f}m height, {np.degrees(tractor.camera_pitch):.1f}° pitch")
    
    # Test pixel projections
    test_points = [
        (960, 540, "Center"),           # Image center
        (960, 1080, "Bottom-center"),   # Bottom center (ground plane ahead)
        (1920, 540, "Right edge"),      # Right edge
        (0, 540, "Left edge"),          # Left edge
    ]
    
    logger.info("\nPixel to 3D Projection Test:")
    for px, py, label in test_points:
        pos_3d = pov.pixel_to_3d_ground_plane(px, py)
        if pos_3d:
            x, y, z = pos_3d
            dist = np.sqrt(x**2 + z**2)
            logger.info(f"  [{label:20s}] Pixel ({px:4d}, {py:4d}) → "
                       f"Position ({x:6.2f}, {y:6.2f}, {z:6.2f})m, Distance: {dist:6.2f}m")
        else:
            logger.info(f"  [{label:20s}] Pixel ({px:4d}, {py:4d}) → Projection failed")
    
    logger.info("\nSafety Zone Configuration:")
    zone = pov.compute_safety_zone()
    logger.info(f"  Front radius: {tractor.safety_radius_front}m")
    logger.info(f"  Side radius: {tractor.safety_radius_side}m")
    
    # Test safety zone
    test_positions = [
        (0, 0.5, "3m ahead, centerline"),
        (0, 1.5, "1.5m ahead, centerline"),
        (0, 2.0, "Front danger zone"),
        (1.0, 0.5, "3m ahead, 1m left"),
        (2.0, 0.5, "3m ahead, 2m left (edge)"),
    ]
    
    logger.info("\nSafety Zone Tests:")
    for x, z, label in test_positions:
        in_zone = pov.is_in_safety_zone(x, z)
        dist = pov.point_to_safety_zone_distance(x, z)
        status = "DANGER" if in_zone else "SAFE"
        logger.info(f"  [{label:35s}] ({x:5.2f}m, {z:5.2f}m) → {status:6s} (Δ: {dist:6.2f}m)")


if __name__ == "__main__":
    test_geometry()
