"""
Context-Aware Agricultural Safety Risk Assessment

Combines:
- Realistic tractor geometry & camera FOV
- Terrain/soil analysis
- Person detection & trajectory tracking
- Physics-based movement modeling
- Multi-factor risk stratification

REALISM: Risk is computed from actual geometric proximity, soil conditions,
movement capability, and tractor dynamics rather than arbitrary thresholds.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict
import logging

from tractor_geometry import TractorPOVGeometry, TractorGeometry, CameraIntrinsics
from terrain_analysis import TerrainAnalysis, TerrainAnalyzer, SoilType, TerrainFormation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk stratification levels."""
    CRITICAL = "CRITICAL"          # Immediate collision risk
    HIGH_WARNING = "HIGH_WARNING"   # Urgent danger
    WARNING = "WARNING"             # Caution required
    LOW_WARNING = "LOW_WARNING"     # Monitor closely
    SAFE = "SAFE"                   # Normal operation


@dataclass
class RiskFactors:
    """Individual risk components."""
    geometric_distance_m: float         # Actual distance to person (m)
    relative_velocity_mps: float        # Approach speed (m/s, >0 = closing)
    time_to_contact_s: Optional[float]  # Predicted collision time (s) or None
    movement_difficulty: float          # 0-1, how hard person can move away
    person_escape_velocity_mps: float   # Estimated speed person can achieve
    tractor_braking_distance_m: float   # Distance to stop (depends on speed)
    soil_slipperiness: float           # 0-1, how slippery (affects braking)
    visibility: float                   # 0-1, how clearly person is visible
    reaction_time_s: float             # Operator+system response time (0.5-1.5s)
    redundancy_factor: float           # 0-1, functional safety redundancy
    
    def __str__(self) -> str:
        ttc_str = f"{self.time_to_contact_s:.2f}s" if self.time_to_contact_s else "N/A"
        return (
            f"Distance: {self.geometric_distance_m:.2f}m | "
            f"Rel.Vel: {self.relative_velocity_mps:.2f}m/s | "
            f"TTC: {ttc_str} | "
            f"Escape: {self.person_escape_velocity_mps:.2f}m/s | "
            f"Difficulty: {self.movement_difficulty:.2f}"
        )


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    risk_level: RiskLevel
    risk_score: float  # 0-1, numerical score
    factors: RiskFactors
    rationale: str
    recommended_action: str
    safety_margin_m: float  # Margin between person and danger zone


class ContextAwareRiskAssessor:
    """
    Realistic risk assessment combining multiple factors.
    """
    
    def __init__(
        self,
        pov: TractorPOVGeometry,
        tractor_speed_kmh: float = 5.0,
        operator_reaction_time_s: float = 1.0,
        detection_confidence_threshold: float = 0.5,
    ):
        """
        Initialize risk assessor.
        
        Args:
            pov: Tractor POV geometry
            tractor_speed_kmh: Current tractor forward speed (km/h)
            operator_reaction_time_s: Operator response delay (s)
            detection_confidence_threshold: Min confidence for person detection
        """
        self.pov = pov
        self.tractor_speed_mps = tractor_speed_kmh / 3.6  # Convert to m/s
        self.operator_reaction_time = operator_reaction_time_s
        self.detection_confidence_threshold = detection_confidence_threshold
        
        # Physics parameters
        self.gravity = 9.81
        self.max_deceleration = 0.7 * self.gravity  # 70% of g (realistic harvester)
        self.max_lateral_accel = 0.5 * self.gravity  # Limited by tires
        
        self.terrain_analyzer = TerrainAnalyzer()
        logger.info(f"Risk assessor initialized: speed {tractor_speed_kmh}km/h, reaction {operator_reaction_time_s}s")
    
    def compute_braking_distance(
        self,
        initial_speed_mps: Optional[float] = None,
        soil_friction: float = 0.8,
    ) -> float:
        """
        Compute distance to stop given current speed and soil conditions.
        
        Args:
            initial_speed_mps: Speed in m/s (None = use current tractor speed)
            soil_friction: Friction coefficient (affected by soil type)
        
        Returns:
            Braking distance in meters
        """
        speed = initial_speed_mps if initial_speed_mps is not None else self.tractor_speed_mps
        
        # Deceleration limited by soil friction
        max_decel = self.max_deceleration * soil_friction
        
        # Distance = v^2 / (2*a)
        braking_dist = (speed ** 2) / (2 * max_decel)
        
        return braking_dist
    
    def compute_person_escape_velocity(
        self,
        terrain: TerrainAnalysis,
        age_factor: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Estimate human movement capability on this terrain.
        
        Args:
            terrain: Terrain analysis
            age_factor: 1.0 for young/healthy, <1.0 for reduced capability
        
        Returns:
            (max_speed_mps, reaction_move_speed_mps)
            - max_speed: Peak speed human can achieve
            - reaction_move_speed: Speed during immediate evasion
        """
        # Base speeds for humans (m/s)
        athletic_speed = 4.5  # ~10 mph
        normal_speed = 2.5    # ~5-6 mph
        elderly_speed = 1.5   # ~3-4 mph
        
        # Assume average human
        base_speed = normal_speed * age_factor
        
        # Reduce based on terrain difficulty
        difficulty_factor = 1.0 - terrain.movement_difficulty
        max_speed = base_speed * difficulty_factor
        
        # Emergency evasion speed (lower due to terrain)
        emergency_speed = base_speed * difficulty_factor * 0.7
        
        # Account for soil conditions
        if terrain.soil_type in [SoilType.CLAY, SoilType.SILTY_CLAY]:
            max_speed *= 0.6  # Stuck in mud
            emergency_speed *= 0.5
        
        return max_speed, emergency_speed
    
    def compute_time_to_contact(
        self,
        person_position_m: Tuple[float, float, float],
        person_velocity_mps: Tuple[float, float, float],
        prediction_horizon_s: float = 3.0,
    ) -> Optional[float]:
        """
        Compute time until person collides with tractor danger zone.
        
        Args:
            person_position_m: (x, y, z) in tractor frame
            person_velocity_mps: (vx, vy, vz) velocity vector
            prediction_horizon_s: How far ahead to predict (s)
        
        Returns:
            Time to contact in seconds, or None if no collision predicted
        """
        # Tractor danger zone (simplified as cylinder)
        danger_zone = self.pov.compute_safety_zone()
        
        # Simulate person trajectory
        for t in np.linspace(0, prediction_horizon_s, int(prediction_horizon_s * 30)):
            # Future position
            future_pos = (
                person_position_m[0] + person_velocity_mps[0] * t,
                person_position_m[1] + person_velocity_mps[1] * t,
                person_position_m[2] + person_velocity_mps[2] * t,
            )
            
            # Check if in danger zone
            if self.pov.is_in_safety_zone(future_pos[0], future_pos[2]):
                return t
        
        return None
    
    def assess_detection(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        detection_history: Optional[List[Tuple[float, float, float, float]]] = None,
        image: Optional[np.ndarray] = None,
        terrain: Optional[TerrainAnalysis] = None,
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment for a detected person.
        
        Args:
            bbox: Detection bounding box (x1, y1, x2, y2)
            confidence: Detection confidence 0-1
            detection_history: Previous detections for trajectory
            image: Current image for terrain analysis
            terrain: Pre-computed terrain analysis, or None to re-compute
        
        Returns:
            RiskAssessment object
        """
        # Extract 3D position
        pos_3d_extended = self.pov.pixel_bbox_to_3d_position(bbox)
        if pos_3d_extended is None:
            return RiskAssessment(
                risk_level=RiskLevel.SAFE,
                risk_score=0.0,
                factors=RiskFactors(
                    geometric_distance_m=float('inf'),
                    relative_velocity_mps=0,
                    time_to_contact_s=None,
                    movement_difficulty=0.5,
                    person_escape_velocity_mps=2.5,
                    tractor_braking_distance_m=self.compute_braking_distance(),
                    soil_slipperiness=0.5,
                    visibility=0.0,
                    reaction_time_s=self.operator_reaction_time,
                    redundancy_factor=0.8,
                ),
                rationale="Person projection failed, assuming outside field of view",
                recommended_action="CONTINUE_NORMAL_OPERATION",
                safety_margin_m=float('inf'),
            )
        
        x, y, z, distance_m = pos_3d_extended
        
        # Terrain analysis
        if terrain is None and image is not None:
            x1, y1, x2, y2 = bbox
            roi = (int(x1), int(y1), int(x2), int(y2))
            terrain = self.terrain_analyzer.analyze_image(image, roi)
        
        if terrain is None:
            # Assume average conditions
            terrain = TerrainAnalysis(
                formation=TerrainFormation.FLAT,
                slope_degrees=2.0,
                soil_type=SoilType.LOAM,
                soil_confidence=0.5,
                vegetation_coverage=0.1,
                moisture_level=0.5,
                compaction_level=0.2,
                drainage_quality="moderate",
                hazard_factors=[],
                movement_difficulty=0.3,
            )
        
        # Compute factors
        factors = self._compute_risk_factors(
            distance_m=distance_m,
            position_3d=(x, y, z),
            detection_history=detection_history,
            terrain=terrain,
            detection_confidence=confidence,
        )
        
        # Combine factors into risk score
        risk_score = self._compute_risk_score(factors, terrain)
        
        # Determine risk level
        risk_level = self._classify_risk_level(risk_score, factors)
        
        # Generate rationale and recommendation
        rationale, action = self._generate_response(risk_level, factors, terrain)
        
        # Safety margin
        safety_margin = distance_m - self.pov.tractor.safety_radius_front
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            rationale=rationale,
            recommended_action=action,
            safety_margin_m=safety_margin,
        )
    
    def _compute_risk_factors(
        self,
        distance_m: float,
        position_3d: Tuple[float, float, float],
        detection_history: Optional[List],
        terrain: TerrainAnalysis,
        detection_confidence: float,
    ) -> RiskFactors:
        """Compute individual risk factor components."""
        
        # Velocity estimation from trajectory
        rel_velocity = 0.0
        if detection_history and len(detection_history) >= 2:
            # Estimate velocity from last 2 detections
            dx = position_3d[0] - detection_history[-1][0]
            dz = position_3d[2] - detection_history[-1][2]
            dt = 1 / 30  # Assume 30 FPS
            rel_velocity = -np.sqrt(dx**2 + dz**2) / dt if dt > 0 else 0
            # Negative = approaching
        
        # Person escape velocity
        max_escape_vel, emergency_escape_vel = self.compute_person_escape_velocity(terrain)
        
        # Braking distance
        soil_friction = 0.9 if terrain.soil_type in [SoilType.SAND, SoilType.LOAMY_SAND] else 0.8
        braking_dist = self.compute_braking_distance(soil_friction=soil_friction)
        
        # Time to contact
        person_vel_mps = (rel_velocity, 0, rel_velocity) if rel_velocity != 0 else (0, 0, 0)
        ttc = self.compute_time_to_contact(position_3d, person_vel_mps)
        
        # Visibility (higher confidence = better visibility)
        visibility = min(1.0, detection_confidence * 1.2)
        
        # Soil slipperiness (affects collision likelihood)
        soil_slipperiness = 0.7 if terrain.moisture_level > 0.6 else 0.5
        
        # Redundancy factor (system health)
        redundancy = 0.8 if detection_confidence > 0.7 else 0.6
        
        return RiskFactors(
            geometric_distance_m=distance_m,
            relative_velocity_mps=rel_velocity,
            time_to_contact_s=ttc,
            movement_difficulty=terrain.movement_difficulty,
            person_escape_velocity_mps=emergency_escape_vel,
            tractor_braking_distance_m=braking_dist,
            soil_slipperiness=soil_slipperiness,
            visibility=visibility,
            reaction_time_s=self.operator_reaction_time,
            redundancy_factor=redundancy,
        )
    
    def _compute_risk_score(self, factors: RiskFactors, terrain: TerrainAnalysis) -> float:
        """
        Combine risk factors into single 0-1 score using realistic model.
        
        Key insights:
        - Distance is primary factor
        - TTC (time-to-contact) vs escape time determines criticality
        - Terrain affects both escape capability and sensor reliability
        """
        # Normalize distance (d_safe = 3m, d_danger = 0.3m)
        safe_dist = self.pov.tractor.safety_radius_front
        danger_dist = 0.3
        
        dist_score = 1.0 - np.clip((factors.geometric_distance_m - danger_dist) / (safe_dist - danger_dist), 0, 1)
        
        # TTC score: low TTC = high risk
        if factors.time_to_contact_s is not None:
            escape_time = factors.geometric_distance_m / max(factors.person_escape_velocity_mps, 0.1)
            reaction_avail_time = factors.time_to_contact_s - factors.reaction_time_s
            
            if reaction_avail_time <= 0:
                ttc_score = 1.0  # No time to react
            else:
                ttc_score = max(0, 1.0 - (reaction_avail_time / escape_time))
        else:
            ttc_score = 0.0
        
        # Approach velocity (closing = risk)
        vel_score = min(1.0, max(0, factors.relative_velocity_mps) / 2.0)
        
        # Escape difficulty (hard to move = can't avoid)
        escape_score = factors.movement_difficulty
        
        # Visibility/confidence
        vis_score = 1.0 - factors.visibility
        
        # Terrain hazards
        hazard_factor = 0.1 * len(terrain.hazard_factors)
        
        # Combine with weighting
        combined_score = (
            0.40 * dist_score +         # Distance most important
            0.25 * ttc_score +          # Time-to-contact critical
            0.15 * vel_score +          # Approach speed
            0.10 * escape_score +       # Can they escape?
            0.07 * vis_score +          # Can we see them?
            0.03 * hazard_factor        # Terrain hazards
        )
        
        # Reliability reduction due to terrain
        reliability_factor = (1.0 - 0.2 * (1.0 - factors.redundancy_factor))
        
        return float(np.clip(combined_score * reliability_factor, 0, 1))
    
    def _classify_risk_level(self, risk_score: float, factors: RiskFactors) -> RiskLevel:
        """Classify risk score into risk level."""
        
        # Hard constraints override scoring
        if factors.geometric_distance_m < 0.5:
            return RiskLevel.CRITICAL
        
        if factors.time_to_contact_s is not None and factors.time_to_contact_s < 0.5:
            return RiskLevel.CRITICAL
        
        # Soft thresholds
        if risk_score >= 0.75:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.50:
            return RiskLevel.HIGH_WARNING
        elif risk_score >= 0.30:
            return RiskLevel.WARNING
        elif risk_score >= 0.10:
            return RiskLevel.LOW_WARNING
        else:
            return RiskLevel.SAFE
    
    def _generate_response(self, risk_level: RiskLevel, factors: RiskFactors, terrain: TerrainAnalysis) -> Tuple[str, str]:
        """Generate rationale and recommended action."""
        
        rationales = {
            RiskLevel.CRITICAL: (
                f"IMMINENT COLLISION RISK: Distance {factors.geometric_distance_m:.2f}m, "
                f"TTC {factors.time_to_contact_s or 'N/A'}, "
                f"terrain {terrain.formation.value} with {terrain.movement_difficulty:.0%} escape difficulty"
            ),
            RiskLevel.HIGH_WARNING: (
                f"High danger: {factors.geometric_distance_m:.2f}m away, "
                f"approaching at {factors.relative_velocity_mps:.2f}m/s on {terrain.soil_type.value} soil"
            ),
            RiskLevel.WARNING: (
                f"Caution: Person {factors.geometric_distance_m:.2f}m ahead on {terrain.formation.value} terrain. "
                f"Escape speed limited to {factors.person_escape_velocity_mps:.2f}m/s due to {terrain.soil_type.value} soil."
            ),
            RiskLevel.LOW_WARNING: (
                f"Monitor: Person {factors.geometric_distance_m:.2f}m away. "
                f"Terrain: {terrain.formation.value}, {len(terrain.hazard_factors)} hazards."
            ),
            RiskLevel.SAFE: (
                f"Safe: Person {factors.geometric_distance_m:.2f}m away on {terrain.formation.value}. "
                f"Good escape routes available."
            ),
        }
        
        actions = {
            RiskLevel.CRITICAL: "EMERGENCY_STOP_IMMEDIATE",
            RiskLevel.HIGH_WARNING: "URGENT_DECELERATION_AND_ALARM",
            RiskLevel.WARNING: "REDUCE_SPEED_AND_ALERT",
            RiskLevel.LOW_WARNING: "MONITOR_CLOSELY_REDUCED_SPEED",
            RiskLevel.SAFE: "NORMAL_OPERATION",
        }
        
        rationale = rationales.get(risk_level, "Unknown risk state")
        action = actions.get(risk_level, "UNKNOWN_ACTION")
        
        return rationale, action


def test_risk_assessment():
    """Test risk assessor."""
    logger.info("=" * 80)
    logger.info("CONTEXT-AWARE RISK ASSESSMENT TEST")
    logger.info("=" * 80)
    
    from tractor_geometry import create_realistic_camera
    
    # Setup
    tractor = TractorGeometry.default_harvester()
    camera = create_realistic_camera()
    pov = TractorPOVGeometry(tractor, camera)
    
    assessor = ContextAwareRiskAssessor(pov, tractor_speed_kmh=4.0)
    
    # Test scenarios
    test_cases = [
        ("safe", (960, 900), 0.95, "Person far away, to the side"),
        ("warning", (960, 700), 0.92, "Person approaching, 2m away"),
        ("critical", (960, 500), 0.98, "Person very close, 0.8m away"),
        ("edge", (200, 600), 0.85, "Person to extreme right"),
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("RISK ASSESSMENT SCENARIOS")
    logger.info("=" * 80)
    
    for scenario, pixel_pos, conf, description in test_cases:
        logger.info(f"\n--- {scenario.upper()}: {description} ---")
        logger.info(f"Pixel position: {pixel_pos}, Confidence: {conf:.2f}")
        
        bbox = (pixel_pos[0] - 40, pixel_pos[1] - 60, pixel_pos[0] + 40, pixel_pos[1] + 80)
        
        assessment = assessor.assess_detection(bbox, conf)
        
        logger.info(f"Risk Level: {assessment.risk_level.value}")
        logger.info(f"Risk Score: {assessment.risk_score:.3f}")
        logger.info(f"Safety Margin: {assessment.safety_margin_m:.2f}m")
        logger.info(f"Factors: {assessment.factors}")
        logger.info(f"Rationale: {assessment.rationale}")
        logger.info(f"Action: {assessment.recommended_action}")


if __name__ == "__main__":
    test_risk_assessment()
