"""
Terrain & Soil Analysis for Agricultural Safety

Analyzes field conditions using computer vision to extract:
- Land formation (plane, slope, distorted terrain)
- Soil type (clay, sand, loam, etc.)
- Vegetation coverage
- Field drainage patterns
- Risk factors based on soil-machinery interaction

This provides CONTEXT for human movement predictions and risk assessment.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoilType(Enum):
    """Soil classifications affecting traction and worker movement."""
    CLAY = "clay"           # Heavy, sticky, poor drainage
    SILTY_CLAY = "silty_clay"
    CLAY_LOAM = "clay_loam"
    LOAM = "loam"           # Balanced texture
    SILT_LOAM = "silt_loam"
    SANDY_LOAM = "sandy_loam"
    LOAMY_SAND = "loamy_sand"
    SAND = "sand"           # Light, drains quickly
    SANDY_SILT = "sandy_silt"
    UNKNOWN = "unknown"


class TerrainFormation(Enum):
    """Terrain formation types."""
    FLAT = "flat"                  # <2° slope
    GENTLE_SLOPE = "gentle_slope"  # 2-5° slope
    MODERATE_SLOPE = "moderate_slope"  # 5-15° slope
    STEEP = "steep"                # >15° slope
    UNDULATING = "undulating"      # Variable height
    RIDGED = "ridged"              # Furrows or ridge patterns
    ERODED = "eroded"              # Gully erosion patterns
    COMPACTED = "compacted"        # Tractor wheel compaction patterns


@dataclass
class TerrainAnalysis:
    """Results of terrain analysis."""
    formation: TerrainFormation
    slope_degrees: float
    soil_type: SoilType
    soil_confidence: float
    vegetation_coverage: float  # 0-1, fraction
    moisture_level: float  # 0-1, inferred from color
    compaction_level: float  # 0-1, wheel compaction visible
    drainage_quality: str  # "good", "moderate", "poor"
    hazard_factors: List[str]  # Lists hazards (mud, stones, roots, etc.)
    movement_difficulty: float  # 0-1, how hard for human to move
    
    def __str__(self) -> str:
        return (
            f"Terrain: {self.formation.value}\n"
            f"  Slope: {self.slope_degrees:.1f}°\n"
            f"  Soil: {self.soil_type.value} (conf: {self.soil_confidence:.2f})\n"
            f"  Vegetation: {self.vegetation_coverage*100:.0f}%\n"
            f"  Moisture: {self.moisture_level:.2f}\n"
            f"  Compaction: {self.compaction_level:.2f}\n"
            f"  Drainage: {self.drainage_quality}\n"
            f"  Hazards: {', '.join(self.hazard_factors)}\n"
            f"  Movement difficulty: {self.movement_difficulty:.2f}"
        )


class TerrainAnalyzer:
    """
    Analyzes terrain from image using CV techniques.
    """
    
    def __init__(self):
        """Initialize terrain analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_image(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> TerrainAnalysis:
        """
        Analyze terrain from image.
        
        Args:
            image: BGR image from harvester camera
            roi: Region of interest (x1, y1, x2, y2) or None for full image
        
        Returns:
            TerrainAnalysis object with extracted features
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            image = image[y1:y2, x1:x2]
        
        # Extract features
        formation, slope = self._analyze_formation(image)
        soil_type, soil_conf = self._classify_soil_type(image)
        veg_coverage = self._compute_vegetation_coverage(image)
        moisture = self._estimate_moisture(image)
        compaction = self._detect_compaction(image)
        drainage = self._assess_drainage(soil_type, moisture, formation)
        hazards = self._identify_hazards(image, soil_type)
        movement_diff = self._compute_movement_difficulty(image, soil_type, formation)
        
        return TerrainAnalysis(
            formation=formation,
            slope_degrees=slope,
            soil_type=soil_type,
            soil_confidence=soil_conf,
            vegetation_coverage=veg_coverage,
            moisture_level=moisture,
            compaction_level=compaction,
            drainage_quality=drainage,
            hazard_factors=hazards,
            movement_difficulty=movement_diff,
        )
    
    def _analyze_formation(self, image: np.ndarray) -> Tuple[TerrainFormation, float]:
        """
        Analyze terrain formation (flat, sloped, undulating, etc).
        
        Uses edge detection + perspective cues.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal lines (flat terrain)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        horizontal_lines = 0
        diagonal_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle)
                
                # Check if roughly horizontal
                if abs(angle_deg) < 15 or abs(angle_deg) > 165:
                    horizontal_lines += 1
                # Check for diagonal (slope indicators)
                elif 20 < angle_deg < 60 or -60 < angle_deg < -20:
                    diagonal_lines += 1
        
        # Vanishing point analysis (perspective cues for slope)
        h, w = gray.shape
        lower_half = edges[h//2:, :]
        
        # Compute horizontal gradient in lower image (perspective convergence)
        gradient_bottom = np.sum(np.abs(np.gradient(lower_half, axis=1)), axis=0)
        center_concentration = np.sum(gradient_bottom[w//4:3*w//4])
        edge_concentration = np.sum(gradient_bottom[0:w//4]) + np.sum(gradient_bottom[3*w//4:])
        
        # Perspective concentration (higher = more slope)
        if center_concentration + edge_concentration > 0:
            perspective_ratio = center_concentration / (center_concentration + edge_concentration)
        else:
            perspective_ratio = 0.5
        
        # Classify formation
        if horizontal_lines > diagonal_lines * 2 and perspective_ratio < 0.4:
            formation = TerrainFormation.FLAT
            slope = 0.5
        elif diagonal_lines > horizontal_lines and perspective_ratio > 0.6:
            formation = TerrainFormation.MODERATE_SLOPE
            slope = 8.0
        elif perspective_ratio > 0.7:
            formation = TerrainFormation.STEEP
            slope = 20.0
        else:
            formation = TerrainFormation.GENTLE_SLOPE
            slope = 3.0
        
        # Detect undulation (varying heights)
        freq_domain = np.abs(np.fft.fft(np.mean(edges, axis=1)))
        undulation_energy = np.sum(freq_domain[5:20])
        if undulation_energy > np.sum(freq_domain) * 0.1:
            formation = TerrainFormation.UNDULATING
        
        return formation, slope
    
    def _classify_soil_type(self, image: np.ndarray) -> Tuple[SoilType, float]:
        """
        Classify soil type from color and texture.
        
        Heuristics:
        - Clay: Dark brown/gray, low texture, high saturation
        - Sand: Light brown, high texture grain, low saturation
        - Loam: Medium brown, balanced texture
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Compute statistics
        h_mean, h_std = np.mean(h), np.std(h)
        s_mean, s_std = np.mean(s), np.std(s)
        v_mean, v_std = np.mean(v), np.std(v)
        
        # Texture analysis via Laplacian variance (sharpness)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Local descriptor histogram
        orb = cv2.ORB_create(nfeatures=100)
        kp, des = orb.detectAndCompute(gray, None)
        
        # Classification logic
        scores = {}
        
        # Clay indicators: dark, low saturation, smooth texture
        clay_score = (1 - v_mean/255) * (1 - s_mean/255) * max(0, 1 - texture_variance/1000)
        scores[SoilType.CLAY] = clay_score
        
        # Sand indicators: light, high saturation, rough texture
        sand_score = (v_mean/255) * (1 - s_mean/255) * min(1, texture_variance/1000)
        scores[SoilType.SAND] = sand_score
        
        # Loam indicators: medium values, balanced
        loam_score = (abs(v_mean/255 - 0.5) * abs(s_mean/255 - 0.5)) * (1 - abs(texture_variance/500 - 0.5))
        scores[SoilType.LOAM] = loam_score
        
        # Silty clay: dark, medium saturation
        silty_clay_score = (1 - v_mean/255) * (0.4 < s_mean/255 < 0.7) * (1 - texture_variance/1500)
        scores[SoilType.SILTY_CLAY] = silty_clay_score
        
        # Sandy loam: medium color, coarse texture
        sandy_loam_score = (0.3 < v_mean/255 < 0.7) * (s_mean/255 * 0.3) * min(1, texture_variance/800)
        scores[SoilType.SANDY_LOAM] = sandy_loam_score
        
        # Find max score
        soil_type = max(scores, key=scores.get)
        confidence = scores[soil_type] / (sum(scores.values()) + 1e-6)
        
        return soil_type, confidence
    
    def _compute_vegetation_coverage(self, image: np.ndarray) -> float:
        """
        Compute vegetation coverage using color-based NDVI approximation.
        """
        b, g, r = cv2.split(image)
        
        # Simple NDVI-like: (green - red) / (green + red + 1)
        ndvi = (g.astype(float) - r.astype(float)) / (g.astype(float) + r.astype(float) + 1)
        
        # Vegetation pixels: NDVI > 0.1
        veg_pixels = np.sum(ndvi > 0.1)
        total_pixels = ndvi.size
        
        coverage = veg_pixels / total_pixels
        return float(coverage)
    
    def _estimate_moisture(self, image: np.ndarray) -> float:
        """
        Estimate soil moisture from darkness/saturation.
        
        Wet soil appears darker and more saturated.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Darkness (value channel)
        darkness = 1 - (v.astype(float) / 255)
        
        # Saturation
        saturation = s.astype(float) / 255
        
        # Moisture score: combination of darkness and saturation
        moisture = (darkness * 0.6 + saturation * 0.4).mean()
        
        return float(np.clip(moisture, 0, 1))
    
    def _detect_compaction(self, image: np.ndarray) -> float:
        """
        Detect tractor wheel compaction patterns.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect linear patterns (wheel tracks)
        # Use morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Difference shows compaction
        diff = cv2.absdiff(gray, morphed)
        
        # Compaction ratio
        compaction = np.mean(diff) / 255.0
        
        return float(np.clip(compaction, 0, 1))
    
    def _assess_drainage(self, soil_type: SoilType, moisture: float, formation: TerrainFormation) -> str:
        """
        Assess drainage quality based on soil type, moisture, and formation.
        """
        # Drainage by soil type
        drainage_scores = {
            SoilType.CLAY: 0.2,  # Poor drainage
            SoilType.SILTY_CLAY: 0.3,
            SoilType.CLAY_LOAM: 0.4,
            SoilType.LOAM: 0.6,
            SoilType.SILT_LOAM: 0.55,
            SoilType.SANDY_LOAM: 0.75,
            SoilType.LOAMY_SAND: 0.85,
            SoilType.SAND: 0.95,  # Good drainage
            SoilType.SANDY_SILT: 0.7,
            SoilType.UNKNOWN: 0.5,
        }
        
        score = drainage_scores[soil_type]
        
        # Formation affects drainage
        if formation == TerrainFormation.STEEP:
            score += 0.15
        elif formation == TerrainFormation.FLAT:
            score -= 0.10
        
        # Moisture overrides
        if moisture > 0.7:
            return "poor"
        elif moisture > 0.5:
            return "moderate"
        else:
            return "good"
    
    def _identify_hazards(self, image: np.ndarray, soil_type: SoilType) -> List[str]:
        """
        Identify specific hazards visible in field.
        """
        hazards = []
        
        # Soil-based hazards
        if soil_type in [SoilType.CLAY, SoilType.SILTY_CLAY]:
            hazards.append("sticky_mud")
        
        if soil_type in [SoilType.SAND, SoilType.LOAMY_SAND]:
            hazards.append("loose_sand")
        
        # Visual hazard detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Dark spots (holes, ruts)
        dark_pixels = np.sum(gray < 80) / gray.size
        if dark_pixels > 0.05:
            hazards.append("ruts_or_holes")
        
        # Bright spots (rocks, stones)
        bright_pixels = np.sum(gray > 200) / gray.size
        if bright_pixels > 0.03:
            hazards.append("stones_or_rocks")
        
        # High variance = uncertain terrain
        if np.std(gray) > 100:
            hazards.append("variable_terrain")
        
        # Vegetation hazards
        veg_cov = self._compute_vegetation_coverage(image)
        if veg_cov > 0.4:
            hazards.append("dense_vegetation")
        
        return hazards
    
    def _compute_movement_difficulty(
        self,
        image: np.ndarray,
        soil_type: SoilType,
        formation: TerrainFormation
    ) -> float:
        """
        Compute how difficult it is for a human to move across this terrain.
        
        0 = easy (flat, dry, compacted)
        1 = very difficult (steep, muddy, rough)
        """
        score = 0.5  # Baseline
        
        # Soil type difficulty
        soil_difficulty = {
            SoilType.CLAY: 0.7,
            SoilType.SILTY_CLAY: 0.6,
            SoilType.CLAY_LOAM: 0.5,
            SoilType.LOAM: 0.4,
            SoilType.SILT_LOAM: 0.4,
            SoilType.SANDY_LOAM: 0.3,
            SoilType.LOAMY_SAND: 0.2,
            SoilType.SAND: 0.4,  # Sand sinks
            SoilType.SANDY_SILT: 0.35,
            SoilType.UNKNOWN: 0.5,
        }
        
        if soil_type in soil_difficulty:
            score = soil_difficulty[soil_type]
        
        # Slope difficulty
        slope_difficulty = {
            TerrainFormation.FLAT: 0.0,
            TerrainFormation.GENTLE_SLOPE: 0.15,
            TerrainFormation.MODERATE_SLOPE: 0.4,
            TerrainFormation.STEEP: 0.8,
            TerrainFormation.UNDULATING: 0.3,
            TerrainFormation.RIDGED: 0.25,
            TerrainFormation.ERODED: 0.6,
            TerrainFormation.COMPACTED: 0.1,
        }
        
        score += slope_difficulty[formation] * 0.3
        
        # Moisture adds difficulty
        moisture = self._estimate_moisture(image)
        score += moisture * 0.2
        
        # Clamp to [0, 1]
        return float(np.clip(score, 0, 1))


def test_terrain_analyzer():
    """Test terrain analyzer."""
    logger.info("=" * 70)
    logger.info("TERRAIN ANALYZER TEST")
    logger.info("=" * 70)
    
    # Create synthetic test images
    test_scenarios = [
        ("flat_dry", np.full((480, 640, 3), [80, 100, 70], dtype=np.uint8)),
        ("hilly_muddy", np.full((480, 640, 3), [50, 60, 40], dtype=np.uint8)),
        ("sandy", np.full((480, 640, 3), [120, 140, 100], dtype=np.uint8)),
    ]
    
    analyzer = TerrainAnalyzer()
    
    for scenario_name, test_image in test_scenarios:
        logger.info(f"\n--- Scenario: {scenario_name} ---")
        analysis = analyzer.analyze_image(test_image)
        logger.info(str(analysis))


if __name__ == "__main__":
    test_terrain_analyzer()
