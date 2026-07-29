"""
LLM-Enhanced Agricultural Safety Risk Assessment

Integrates Large Language Models for intelligent scene understanding,
risk prediction, and safety decision making in agricultural environments.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass
from enum import Enum

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. LLM features will be disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not available. Claude LLM features will be disabled.")

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing without API calls

@dataclass
class SceneDescription:
    """Structured scene description for LLM analysis."""
    num_humans: int
    human_positions: List[Dict[str, Any]]
    movement_patterns: List[str]
    environmental_factors: List[str]
    equipment_status: str
    time_context: str

@dataclass
class LLMAnalysisResult:
    """Results from LLM risk analysis."""
    overall_risk_level: str
    risk_score: float
    reasoning: str
    predicted_scenarios: List[str]
    recommended_actions: List[str]
    confidence_score: float
    processing_time: float

class LLMAgriSafetyAssessor:
    """
    LLM-enhanced risk assessment for agricultural safety scenarios.

    Uses advanced language models to provide intelligent analysis of:
    - Scene understanding and context
    - Risk prediction based on movement patterns
    - Safety decision making
    - Natural language explanations
    """

    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize LLM assessor.

        Args:
            provider: LLM provider to use
            api_key: API key (if None, uses environment variables)
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key(provider)

        # Initialize clients
        if provider == LLMProvider.OPENAI and OPENAI_AVAILABLE:
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            openai.api_key = self.api_key
            self.client = openai.OpenAI()
        elif provider == LLMProvider.ANTHROPIC and ANTHROPIC_AVAILABLE:
            if not self.api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif provider == LLMProvider.MOCK:
            self.client = None
            logging.info("Using mock LLM for testing")
        else:
            raise ValueError(f"Provider {provider} not available or not supported")

        # Risk assessment prompt templates
        self.system_prompt = self._get_system_prompt()
        self.analysis_cache = {}  # Cache for similar scenes

        logging.info(f"LLM Agricultural Safety Assessor initialized with {provider.value} provider")

    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key from environment variables."""
        if provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            return os.getenv("ANTHROPIC_API_KEY")
        return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for agricultural safety analysis."""
        return """You are an expert agricultural safety AI specializing in autonomous harvester operations.

Your expertise includes:
- Understanding agricultural field conditions and equipment
- Analyzing human movement patterns in farming environments
- Predicting potential safety hazards and collision risks
- Providing actionable safety recommendations
- Assessing risk levels based on proximity, speed, and context

You must provide analysis in JSON format with the following structure:
{
    "overall_risk_level": "SAFE|LOW_WARNING|WARNING|HIGH_WARNING|CRITICAL",
    "risk_score": 0.0-1.0,
    "reasoning": "detailed explanation of risk assessment",
    "predicted_scenarios": ["potential future scenarios"],
    "recommended_actions": ["specific safety actions"],
    "confidence_score": 0.0-1.0
}

Consider these agricultural safety factors:
- Harvester speed and operational context
- Human proximity to equipment path
- Movement direction and speed
- Environmental conditions (dust, visibility, terrain)
- Time of day and lighting conditions
- Human behavior patterns in agricultural settings"""

    def analyze_scene(self, scene_description: SceneDescription) -> LLMAnalysisResult:
        """
        Analyze agricultural scene using LLM for enhanced risk assessment.

        Args:
            scene_description: Structured scene information

        Returns:
            LLMAnalysisResult: LLM analysis results
        """
        start_time = time.time()

        # Create cache key for similar scenes
        cache_key = self._create_cache_key(scene_description)

        # Check cache first
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result

        # Create analysis prompt
        user_prompt = self._create_analysis_prompt(scene_description)

        try:
            if self.provider == LLMProvider.MOCK:
                result = self._mock_analysis(scene_description)
            else:
                result = self._call_llm(user_prompt)

            result.processing_time = time.time() - start_time

            # Cache result
            self.analysis_cache[cache_key] = result

            return result

        except Exception as e:
            logging.error(f"LLM analysis failed: {e}")
            # Return safe fallback
            return LLMAnalysisResult(
                overall_risk_level="SAFE",
                risk_score=0.1,
                reasoning=f"LLM analysis failed: {e}. Defaulting to safe assessment.",
                predicted_scenarios=["Unable to predict scenarios due to analysis failure"],
                recommended_actions=["Continue with standard safety protocols"],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )

    def _create_cache_key(self, scene: SceneDescription) -> str:
        """Create cache key for scene similarity."""
        # Simple cache key based on number of humans and movement patterns
        movement_summary = "_".join(sorted(scene.movement_patterns))
        return f"{scene.num_humans}_{movement_summary}_{scene.equipment_status}"

    def _create_analysis_prompt(self, scene: SceneDescription) -> str:
        """Create detailed analysis prompt for the LLM."""
        prompt = f"""Analyze this agricultural safety scenario for autonomous harvester operations:

SCENE DESCRIPTION:
- Number of humans detected: {scene.num_humans}
- Equipment status: {scene.equipment_status}
- Time context: {scene.time_context}
- Environmental factors: {', '.join(scene.environmental_factors)}

HUMAN DETAILS:
"""

        for i, human in enumerate(scene.human_positions):
            prompt += f"""
Human {i+1}:
- Position: {human.get('position', 'unknown')}
- Distance estimate: {human.get('distance_m', 'unknown')} meters
- Movement: {human.get('movement', 'stationary')}
- Risk level: {human.get('current_risk', 'unknown')}
- Bounding box ratio: {human.get('bbox_ratio', 0):.3f}
"""

        prompt += f"""

MOVEMENT PATTERNS: {', '.join(scene.movement_patterns)}

Please analyze this scenario and provide your assessment focusing on:
1. Overall risk to harvester operations and human safety
2. Potential collision scenarios based on current trajectories
3. Recommended safety actions for the autonomous system
4. Confidence in your assessment

Consider agricultural context: harvesters operate at 1-2 m/s, have 2.5m width, and require 5-10m safety zones."""

        return prompt

    def _call_llm(self, prompt: str) -> LLMAnalysisResult:
        """Call the LLM API and parse results."""
        try:
            if self.provider == LLMProvider.OPENAI:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent safety analysis
                    max_tokens=1000
                )
                content = response.choices[0].message.content

            elif self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.1,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            # Parse JSON response
            result_data = json.loads(content)

            return LLMAnalysisResult(
                overall_risk_level=result_data.get("overall_risk_level", "SAFE"),
                risk_score=float(result_data.get("risk_score", 0.0)),
                reasoning=result_data.get("reasoning", "No reasoning provided"),
                predicted_scenarios=result_data.get("predicted_scenarios", []),
                recommended_actions=result_data.get("recommended_actions", []),
                confidence_score=float(result_data.get("confidence_score", 0.5)),
                processing_time=0.0  # Will be set by caller
            )

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            raise
        except Exception as e:
            logging.error(f"LLM API call failed: {e}")
            raise

    def _mock_analysis(self, scene: SceneDescription) -> LLMAnalysisResult:
        """Mock analysis for testing without API calls."""
        # Simple rule-based mock analysis
        high_risk_count = sum(1 for h in scene.human_positions
                            if h.get('distance_m', 100) < 10 and
                            'approaching' in h.get('movement', ''))

        if high_risk_count > 0:
            risk_level = "CRITICAL" if high_risk_count > 1 else "HIGH_WARNING"
            risk_score = 0.8 + (high_risk_count * 0.1)
        elif scene.num_humans > 3:
            risk_level = "WARNING"
            risk_score = 0.6
        elif scene.num_humans > 0:
            risk_level = "LOW_WARNING"
            risk_score = 0.3
        else:
            risk_level = "SAFE"
            risk_score = 0.1

        return LLMAnalysisResult(
            overall_risk_level=risk_level,
            risk_score=min(1.0, risk_score),
            reasoning=f"Mock analysis: {scene.num_humans} humans detected with {high_risk_count} high-risk individuals.",
            predicted_scenarios=[
                "Potential human-equipment interaction",
                "Movement pattern continuation",
                "Environmental factor changes"
            ],
            recommended_actions=[
                "Maintain safe distance",
                "Monitor human trajectories",
                "Prepare emergency stop if needed"
            ],
            confidence_score=0.7,
            processing_time=0.0
        )

    def enhance_risk_assessment(self, traditional_risk_data: Dict[str, Any],
                              scene_description: SceneDescription) -> Dict[str, Any]:
        """
        Enhance traditional risk assessment with LLM analysis.

        Args:
            traditional_risk_data: Results from traditional computer vision risk assessment
            scene_description: Structured scene description

        Returns:
            Enhanced risk assessment combining both approaches
        """
        # Get LLM analysis
        llm_result = self.analyze_scene(scene_description)

        # Combine traditional and LLM assessments
        traditional_score = traditional_risk_data.get('risk_score', 0.0)
        llm_score = llm_result.risk_score

        # Weighted combination (70% traditional, 30% LLM for stability)
        combined_score = (traditional_score * 0.7) + (llm_score * 0.3)

        # Use more conservative risk level (higher of the two)
        risk_levels = ["SAFE", "LOW_WARNING", "WARNING", "HIGH_WARNING", "CRITICAL"]
        trad_level_idx = risk_levels.index(traditional_risk_data.get('risk_level', 'SAFE'))
        llm_level_idx = risk_levels.index(llm_result.overall_risk_level)
        combined_level_idx = max(trad_level_idx, llm_level_idx)
        combined_level = risk_levels[combined_level_idx]

        # Combine reasoning
        combined_reasoning = f"""
Traditional CV Analysis: {traditional_risk_data.get('reasoning', 'N/A')}
LLM Enhanced Analysis: {llm_result.reasoning}
Combined Assessment: Integrated computer vision and AI reasoning for robust safety evaluation.
"""

        return {
            'risk_level': combined_level,
            'risk_score': combined_score,
            'traditional_score': traditional_score,
            'llm_score': llm_score,
            'reasoning': combined_reasoning.strip(),
            'predicted_scenarios': llm_result.predicted_scenarios,
            'recommended_actions': llm_result.recommended_actions,
            'llm_confidence': llm_result.confidence_score,
            'processing_time': llm_result.processing_time,
            'enhanced': True
        }

def create_scene_description(num_humans: int, human_data: List[Dict],
                           movement_patterns: List[str], environmental_factors: List[str] = None) -> SceneDescription:
    """
    Helper function to create scene description from detection data.

    Args:
        num_humans: Number of humans detected
        human_data: List of human detection data
        movement_patterns: List of movement pattern descriptions
        environmental_factors: List of environmental factors

    Returns:
        SceneDescription: Structured scene description
    """
    if environmental_factors is None:
        environmental_factors = ["daylight", "clear_visibility", "standard_field_conditions"]

    return SceneDescription(
        num_humans=num_humans,
        human_positions=human_data,
        movement_patterns=movement_patterns,
        environmental_factors=environmental_factors,
        equipment_status="active_harvesting",
        time_context="field_operation_hours"
    )