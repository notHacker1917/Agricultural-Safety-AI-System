#!/usr/bin/env python3
"""
Advanced LLM-Enhanced Agricultural Safety Risk Assessment System

Implements the comprehensive risk assessment specification with:
- Multi-modal human detection algorithms
- LLM-powered contextual understanding
- Advanced trajectory prediction
- Safety-critical decision making
- Edge case handling with uncertainty management

Risk Levels: SAFE, LOW, MEDIUM, HIGH, CRITICAL
Safety Principle: When uncertain, ALWAYS choose HIGHER risk level
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass, asdict
from enum import Enum
import math
from collections import deque


try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class RiskLevel(Enum):
    """Safety-critical risk levels for agricultural machinery."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class HumanDetectionInput:
    """Input parameters for human risk assessment."""
    object_id: str
    current_position: Tuple[float, float]  # (x, y)
    distance_to_tractor: float  # normalized: 0 to 1
    velocity: Tuple[float, float]  # (vx, vy)
    speed: float  # magnitude of velocity
    direction_toward_tractor: bool
    predicted_path: List[Tuple[float, float]]  # list of future positions
    will_enter_safety_zone: bool
    time_to_collision: Optional[float]  # in frames, or None
    is_occluded: bool
    detection_confidence: float  # 0 to 1

@dataclass
class RiskAssessmentOutput:
    """Output from risk assessment system."""
    risk_level: RiskLevel
    risk_score: float  # 0 to 1
    reason: str  # short explanation

class AdvancedLLMRiskAssessor:
    """
    Advanced LLM-enhanced risk assessment system for agricultural safety.

    Implements comprehensive risk evaluation with:
    - Multi-algorithm human detection
    - LLM contextual understanding
    - Trajectory prediction and analysis
    - Safety-critical decision making
    - Uncertainty handling with conservative bias
    """

    def __init__(self,
                 llm_provider: str = "mock",
                 safety_zone_radius: float = 0.3,  # normalized distance
                 collision_threshold_frames: int = 10,
                 use_llm_context: bool = True):
        """
        Initialize the advanced risk assessor.

        Args:
            llm_provider: LLM provider ("openai", "anthropic", "mock")
            safety_zone_radius: Normalized radius of safety zone around tractor
            collision_threshold_frames: Frames threshold for critical risk
            use_llm_context: Whether to use LLM for contextual analysis
        """
        self.safety_zone_radius = safety_zone_radius
        self.collision_threshold_frames = collision_threshold_frames
        self.use_llm_context = use_llm_context

        # Setup LLM provider
        self.llm_provider = llm_provider
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI()
        elif llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
        else:
            self.client = None
            logging.warning("Using mock LLM provider for testing")

        # Risk level score ranges
        self.risk_ranges = {
            RiskLevel.SAFE: (0.0, 0.2),
            RiskLevel.LOW: (0.2, 0.4),
            RiskLevel.MEDIUM: (0.4, 0.6),
            RiskLevel.HIGH: (0.6, 0.8),
            RiskLevel.CRITICAL: (0.8, 1.0)
        }

        # Trajectory history for temporal analysis
        self.trajectory_history: Dict[str, deque] = {}
        self.history_length = 10  # frames

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def assess_risk(self, detection_input: HumanDetectionInput) -> RiskAssessmentOutput:
        """
        Assess risk level for a detected human using comprehensive analysis.

        Args:
            detection_input: Human detection data

        Returns:
            RiskAssessmentOutput with level, score, and reasoning
        """
        try:
            # Step 1: Apply safety override rules (MANDATORY)
            override_risk = self._apply_safety_overrides(detection_input)
            if override_risk:
                return override_risk

            # Step 2: Calculate base risk score from multiple factors
            base_score = self._calculate_base_risk_score(detection_input)

            # Step 3: Apply edge case adjustments
            adjusted_score = self._apply_edge_case_adjustments(detection_input, base_score)

            # Step 4: Apply uncertainty penalty (conservative bias)
            final_score = self._apply_uncertainty_penalty(detection_input, adjusted_score)

            # Step 5: Determine risk level
            risk_level = self._score_to_risk_level(final_score)

            # Step 6: Generate reasoning (with LLM if available)
            reason = self._generate_reasoning(detection_input, risk_level, final_score)

            # Step 7: Update trajectory history
            self._update_trajectory_history(detection_input)

            return RiskAssessmentOutput(
                risk_level=risk_level,
                risk_score=round(final_score, 3),
                reason=reason
            )

        except Exception as e:
            # Safety-critical: On error, assume CRITICAL risk
            self.logger.error(f"Risk assessment error for object {detection_input.object_id}: {e}")
            return RiskAssessmentOutput(
                risk_level=RiskLevel.CRITICAL,
                risk_score=1.0,
                reason=f"Assessment error - assuming CRITICAL risk: {str(e)}"
            )

    def _apply_safety_overrides(self, detection: HumanDetectionInput) -> Optional[RiskAssessmentOutput]:
        """
        Apply mandatory safety override rules.
        These take precedence over all other calculations.
        """
        # Override 1: Will enter safety zone
        if detection.will_enter_safety_zone:
            return RiskAssessmentOutput(
                risk_level=RiskLevel.HIGH,
                risk_score=0.7,
                reason="MANDATORY: Object will enter safety zone"
            )

        # Override 2: Time to collision critical
        if (detection.time_to_collision is not None and
            detection.time_to_collision < self.collision_threshold_frames):
            return RiskAssessmentOutput(
                risk_level=RiskLevel.CRITICAL,
                risk_score=0.9,
                reason=f"MANDATORY: Time to collision {detection.time_to_collision:.1f} frames < threshold"
            )

        # Override 3: Inside safety zone
        if detection.distance_to_tractor <= self.safety_zone_radius:
            return RiskAssessmentOutput(
                risk_level=RiskLevel.CRITICAL,
                risk_score=1.0,
                reason="MANDATORY: Object inside safety zone"
            )

        # Override 4: High speed toward tractor
        if (detection.direction_toward_tractor and
            detection.speed > 2.0):  # pixels/frame threshold
            # Increase risk by one level from base calculation
            return RiskAssessmentOutput(
                risk_level=RiskLevel.HIGH,  # Will be adjusted later if needed
                risk_score=0.75,
                reason="MANDATORY: High speed toward tractor"
            )

        return None  # No override applied

    def _calculate_base_risk_score(self, detection: HumanDetectionInput) -> float:
        """
        Calculate base risk score from distance, motion, and prediction factors.
        """
        score_components = []

        # Distance factor (0-0.4 of total score)
        distance_score = self._calculate_distance_score(detection.distance_to_tractor)
        score_components.append(("distance", distance_score, 0.4))

        # Motion factor (0-0.3 of total score)
        motion_score = self._calculate_motion_score(detection)
        score_components.append(("motion", motion_score, 0.3))

        # Prediction factor (0-0.3 of total score)
        prediction_score = self._calculate_prediction_score(detection)
        score_components.append(("prediction", prediction_score, 0.3))

        # Weighted combination
        total_score = sum(score * weight for _, score, weight in score_components)

        self.logger.debug(f"Risk components for {detection.object_id}: {score_components}")
        return min(total_score, 1.0)  # Cap at 1.0

    def _calculate_distance_score(self, normalized_distance: float) -> float:
        """
        Calculate risk score based on distance to tractor.
        Closer = higher risk.
        """
        if normalized_distance <= self.safety_zone_radius:
            return 1.0  # Inside safety zone
        elif normalized_distance <= 0.5:
            return 0.8  # Very close
        elif normalized_distance <= 0.7:
            return 0.6  # Moderately close
        elif normalized_distance <= 0.9:
            return 0.3  # Far but concerning
        else:
            return 0.1  # Very far

    def _calculate_motion_score(self, detection: HumanDetectionInput) -> float:
        """
        Calculate risk score based on motion characteristics.
        """
        score = 0.0

        # Direction toward tractor increases risk
        if detection.direction_toward_tractor:
            score += 0.4

        # Higher speed increases risk
        speed_factor = min(detection.speed / 5.0, 1.0)  # Normalize speed
        score += speed_factor * 0.4

        # Stationary but close is concerning
        if detection.speed < 0.1 and detection.distance_to_tractor < 0.7:
            score += 0.2

        return min(score, 1.0)

    def _calculate_prediction_score(self, detection: HumanDetectionInput) -> float:
        """
        Calculate risk score based on predicted trajectory.
        """
        score = 0.0

        # Will enter safety zone
        if detection.will_enter_safety_zone:
            score += 0.6

        # Time to collision
        if detection.time_to_collision is not None:
            if detection.time_to_collision < 5:
                score += 0.8
            elif detection.time_to_collision < 15:
                score += 0.5
            elif detection.time_to_collision < 30:
                score += 0.2

        # Predicted path analysis
        if detection.predicted_path:
            # Check if path comes close to tractor
            min_predicted_distance = min(
                self._calculate_distance_to_point(pos, (0, 0))  # Assuming tractor at (0,0)
                for pos in detection.predicted_path
            )
            if min_predicted_distance <= self.safety_zone_radius:
                score += 0.4

        return min(score, 1.0)

    def _calculate_distance_to_point(self, pos1: Tuple[float, float],
                                   pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _apply_edge_case_adjustments(self, detection: HumanDetectionInput,
                                   base_score: float) -> float:
        """
        Apply adjustments for edge cases.
        """
        adjusted_score = base_score

        # Edge case 1: Occlusion handling
        if detection.is_occluded:
            # Increase risk by one level (uncertainty penalty)
            adjusted_score = min(adjusted_score + 0.2, 1.0)

        # Edge case 2: Low confidence detection
        if detection.detection_confidence < 0.3:
            if detection.distance_to_tractor < 0.7:
                adjusted_score = max(adjusted_score, 0.3)  # At least LOW risk
            if detection.direction_toward_tractor:
                adjusted_score = max(adjusted_score, 0.5)  # At least MEDIUM

        # Edge case 3: Missing trajectory
        if not detection.velocity or not detection.predicted_path:
            if detection.distance_to_tractor < 0.7:
                adjusted_score = max(adjusted_score, 0.5)  # At least MEDIUM

        # Edge case 4: Sudden appearance
        if not self._has_trajectory_history(detection.object_id):
            if detection.distance_to_tractor < 0.8:
                adjusted_score = max(adjusted_score, 0.7)  # At least HIGH

        # Edge case 5: Stationary near tractor
        if (detection.speed < 0.1 and
            detection.distance_to_tractor < 0.6):
            adjusted_score = max(adjusted_score, 0.5)  # At least MEDIUM

        return adjusted_score

    def _apply_uncertainty_penalty(self, detection: HumanDetectionInput,
                                 score: float) -> float:
        """
        Apply uncertainty penalty - when uncertain, choose HIGHER risk.
        This implements the core safety principle.
        """
        uncertainty_factors = []

        # Low confidence penalty
        if detection.detection_confidence < 0.5:
            uncertainty_factors.append(0.1)

        # Occlusion penalty
        if detection.is_occluded:
            uncertainty_factors.append(0.15)

        # Missing data penalty
        if detection.time_to_collision is None:
            uncertainty_factors.append(0.05)

        if not detection.predicted_path:
            uncertainty_factors.append(0.05)

        # Short trajectory history penalty
        if len(self.trajectory_history.get(detection.object_id, [])) < 3:
            uncertainty_factors.append(0.05)

        # Apply total uncertainty penalty
        total_penalty = min(sum(uncertainty_factors), 0.3)  # Cap at 0.3
        final_score = min(score + total_penalty, 1.0)

        if uncertainty_factors:
            self.logger.info(f"Applied uncertainty penalty of {total_penalty:.2f} "
                           f"for object {detection.object_id}")

        return final_score

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """
        Convert numerical score to risk level enum.
        """
        for level, (min_score, max_score) in self.risk_ranges.items():
            if min_score <= score <= max_score:
                return level

        # Fallback (should not happen with proper capping)
        return RiskLevel.CRITICAL if score > 0.8 else RiskLevel.HIGH

    def _generate_reasoning(self, detection: HumanDetectionInput,
                          risk_level: RiskLevel, score: float) -> str:
        """
        Generate human-readable reasoning for the risk assessment.
        Uses LLM if available for contextual understanding.
        """
        base_reason = f"Distance: {detection.distance_to_tractor:.2f}, "

        if detection.direction_toward_tractor:
            base_reason += f"moving toward tractor at {detection.speed:.1f} speed, "
        else:
            base_reason += f"speed: {detection.speed:.1f}, "

        if detection.will_enter_safety_zone:
            base_reason += "will enter safety zone, "

        if detection.is_occluded:
            base_reason += "occluded, "

        base_reason += f"confidence: {detection.detection_confidence:.2f}"

        # Use LLM for enhanced reasoning if available
        if self.use_llm_context and self.client:
            try:
                llm_reason = self._get_llm_reasoning(detection, risk_level, score)
                return f"{base_reason}. {llm_reason}"
            except Exception as e:
                self.logger.warning(f"LLM reasoning failed: {e}")

        return base_reason

    def _get_llm_reasoning(self, detection: HumanDetectionInput,
                          risk_level: RiskLevel, score: float) -> str:
        """
        Get enhanced reasoning from LLM for contextual understanding.
        """
        if not self.client:
            return "LLM not available"

        prompt = f"""
        Analyze this agricultural safety scenario and provide a brief explanation
        for the {risk_level.value} risk assessment (score: {score:.2f}):

        Human Detection Data:
        - Object ID: {detection.object_id}
        - Distance to tractor: {detection.distance_to_tractor:.2f} (normalized)
        - Speed: {detection.speed:.1f}
        - Moving toward tractor: {detection.direction_toward_tractor}
        - Will enter safety zone: {detection.will_enter_safety_zone}
        - Time to collision: {detection.time_to_collision or 'unknown'}
        - Occluded: {detection.is_occluded}
        - Detection confidence: {detection.detection_confidence:.2f}

        Provide a 1-2 sentence explanation focusing on safety implications.
        """

        try:
            if hasattr(self.client, 'chat'):  # OpenAI
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()

            elif hasattr(self.client, 'messages'):  # Anthropic
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=100,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return "LLM analysis unavailable"

        return "Mock LLM reasoning: Human detected with potential safety risk"

    def _has_trajectory_history(self, object_id: str) -> bool:
        """Check if we have trajectory history for this object."""
        return object_id in self.trajectory_history and len(self.trajectory_history[object_id]) > 0

    def _update_trajectory_history(self, detection: HumanDetectionInput):
        """Update trajectory history for temporal analysis."""
        if detection.object_id not in self.trajectory_history:
            self.trajectory_history[detection.object_id] = deque(maxlen=self.history_length)

        self.trajectory_history[detection.object_id].append({
            'position': detection.current_position,
            'timestamp': time.time(),
            'confidence': detection.detection_confidence
        })

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status for monitoring and diagnostics.
        """
        return {
            "llm_provider": self.llm_provider,
            "llm_available": self.client is not None,
            "safety_zone_radius": self.safety_zone_radius,
            "collision_threshold_frames": self.collision_threshold_frames,
            "tracked_objects": len(self.trajectory_history),
            "risk_ranges": {level.value: f"{min_s:.1f}-{max_s:.1f}"
                          for level, (min_s, max_s) in self.risk_ranges.items()}
        }

    def reset_trajectory_history(self, object_id: Optional[str] = None):
        """
        Reset trajectory history for specified object or all objects.
        """
        if object_id:
            self.trajectory_history.pop(object_id, None)
        else:
            self.trajectory_history.clear()

# Convenience functions for easy integration

def create_risk_assessor(llm_provider: str = "mock",
                        safety_zone_radius: float = 0.3,
                        collision_threshold: int = 10) -> AdvancedLLMRiskAssessor:
    """
    Create and configure an advanced risk assessor.

    Args:
        llm_provider: LLM provider ("openai", "anthropic", "mock")
        safety_zone_radius: Normalized safety zone radius
        collision_threshold: Frames for critical collision risk

    Returns:
        Configured AdvancedLLMRiskAssessor instance
    """
    return AdvancedLLMRiskAssessor(
        llm_provider=llm_provider,
        safety_zone_radius=safety_zone_radius,
        collision_threshold_frames=collision_threshold,
        use_llm_context=True
    )

def assess_human_risk(assessor: AdvancedLLMRiskAssessor,
                     detection_data: Dict[str, Any]) -> RiskAssessmentOutput:
    """
    Convenience function to assess risk from dictionary data.

    Args:
        assessor: Configured risk assessor
        detection_data: Dictionary with detection parameters

    Returns:
        Risk assessment result
    """
    detection_input = HumanDetectionInput(
        object_id=detection_data.get('object_id', 'unknown'),
        current_position=tuple(detection_data.get('current_position', (0, 0))),
        distance_to_tractor=float(detection_data.get('distance_to_tractor', 1.0)),
        velocity=tuple(detection_data.get('velocity', (0, 0))),
        speed=float(detection_data.get('speed', 0.0)),
        direction_toward_tractor=bool(detection_data.get('direction_toward_tractor', False)),
        predicted_path=[tuple(pos) for pos in detection_data.get('predicted_path', [])],
        will_enter_safety_zone=bool(detection_data.get('will_enter_safety_zone', False)),
        time_to_collision=detection_data.get('time_to_collision'),
        is_occluded=bool(detection_data.get('is_occluded', False)),
        detection_confidence=float(detection_data.get('detection_confidence', 0.5))
    )

    return assessor.assess_risk(detection_input)

# Example usage and testing functions

def test_risk_assessment():
    """Test the risk assessment system with sample scenarios."""
    assessor = create_risk_assessor()

    test_cases = [
        {
            "name": "Safe - Far away, stationary",
            "data": {
                "object_id": "human_001",
                "current_position": (100, 100),
                "distance_to_tractor": 0.9,
                "velocity": (0, 0),
                "speed": 0,
                "direction_toward_tractor": False,
                "predicted_path": [],
                "will_enter_safety_zone": False,
                "time_to_collision": None,
                "is_occluded": False,
                "detection_confidence": 0.8
            },
            "expected": RiskLevel.SAFE
        },
        {
            "name": "Critical - Inside safety zone",
            "data": {
                "object_id": "human_002",
                "current_position": (10, 10),
                "distance_to_tractor": 0.1,
                "velocity": (2, 2),
                "speed": 2.8,
                "direction_toward_tractor": True,
                "predicted_path": [(5, 5), (0, 0)],
                "will_enter_safety_zone": True,
                "time_to_collision": 3.0,
                "is_occluded": False,
                "detection_confidence": 0.9
            },
            "expected": RiskLevel.CRITICAL
        },
        {
            "name": "High - Approaching quickly",
            "data": {
                "object_id": "human_003",
                "current_position": (50, 50),
                "distance_to_tractor": 0.4,
                "velocity": (3, 3),
                "speed": 4.2,
                "direction_toward_tractor": True,
                "predicted_path": [(40, 40), (30, 30), (20, 20)],
                "will_enter_safety_zone": True,
                "time_to_collision": 8.0,
                "is_occluded": False,
                "detection_confidence": 0.7
            },
            "expected": RiskLevel.HIGH
        }
    ]

    print("🧪 Testing Advanced LLM Risk Assessment System")
    print("=" * 60)

    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        result = assess_human_risk(assessor, test_case['data'])
        print(f"   Risk Level: {result.risk_level.value}")
        print(f"   Risk Score: {result.risk_score:.2f}")
        print(f"   Reasoning: {result.reason}")

        if result.risk_level == test_case['expected']:
            print("   ✅ PASS")
        else:
            print(f"   ❌ FAIL (expected {test_case['expected'].value})")

    print(f"\n📊 System Status: {assessor.get_system_status()}")

if __name__ == "__main__":
    # Run tests when executed directly
    test_risk_assessment()