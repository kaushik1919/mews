"""
Ensemble Weights Configuration.

MEWS-FIN Phase 4.2: Explicit weights for combining heuristic and ML scores.

Key Constraints (NON-NEGOTIABLE):
    - w_heuristic + w_ml = 1.0
    - w_heuristic >= 0.30 (interpretability anchor)
    - w_ml <= 0.70 (bounded ML influence)

Model Selection Rationale:
    - Primary ML model: Random Forest (best validation accuracy, stable)
    - Secondary ML model: XGBoost (if available, for diversity)
    - Heuristic always included (interpretability guarantee)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelRole(Enum):
    """Role of a model in the ensemble."""

    HEURISTIC = "heuristic"  # Always included, interpretability anchor
    PRIMARY_ML = "primary_ml"  # Best performing ML model
    SECONDARY_ML = "secondary_ml"  # Optional diversity model


@dataclass
class ModelWeight:
    """Weight configuration for a single model."""

    model_name: str
    role: ModelRole
    weight: float
    justification: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")


@dataclass
class EnsembleWeights:
    """
    Complete ensemble weight configuration.

    Constraints enforced at initialization:
        - Total weights sum to 1.0
        - Heuristic weight >= 0.30
        - ML weights combined <= 0.70
    """

    heuristic_weight: float
    primary_ml_weight: float
    secondary_ml_weight: float = 0.0

    # Model identifiers
    primary_ml_model: str = "random_forest"
    secondary_ml_model: str | None = None

    # Justifications
    heuristic_justification: str = "Interpretability anchor; stable baseline"
    primary_ml_justification: str = "Best validation accuracy among ML models"
    secondary_ml_justification: str = ""

    def __post_init__(self) -> None:
        """Validate weight constraints."""
        # Check heuristic minimum
        if self.heuristic_weight < 0.30:
            raise ValueError(
                f"Heuristic weight must be >= 0.30 for interpretability, "
                f"got {self.heuristic_weight}"
            )

        # Check ML maximum
        ml_total = self.primary_ml_weight + self.secondary_ml_weight
        if ml_total > 0.70:
            raise ValueError(
                f"Combined ML weight must be <= 0.70, got {ml_total}"
            )

        # Check sum to 1
        total = self.heuristic_weight + ml_total
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total}"
            )

        # Validate secondary model
        if self.secondary_ml_weight > 0 and not self.secondary_ml_model:
            raise ValueError(
                "secondary_ml_model must be specified if secondary_ml_weight > 0"
            )

    @property
    def total_ml_weight(self) -> float:
        """Total weight assigned to ML models."""
        return self.primary_ml_weight + self.secondary_ml_weight

    def get_model_weights(self) -> dict[str, float]:
        """Get all model weights as a dictionary."""
        weights = {
            "heuristic": self.heuristic_weight,
            self.primary_ml_model: self.primary_ml_weight,
        }
        if self.secondary_ml_model and self.secondary_ml_weight > 0:
            weights[self.secondary_ml_model] = self.secondary_ml_weight
        return weights

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "heuristic_weight": self.heuristic_weight,
            "primary_ml_weight": self.primary_ml_weight,
            "secondary_ml_weight": self.secondary_ml_weight,
            "primary_ml_model": self.primary_ml_model,
            "secondary_ml_model": self.secondary_ml_model,
            "heuristic_justification": self.heuristic_justification,
            "primary_ml_justification": self.primary_ml_justification,
            "secondary_ml_justification": self.secondary_ml_justification,
            "total_ml_weight": self.total_ml_weight,
        }


# ==============================================================================
# DEFAULT ENSEMBLE WEIGHTS
# ==============================================================================
# These weights are the result of Phase 4.1 model comparison.
#
# Selection Rationale:
# - Heuristic (0.35): Provides interpretable baseline, stable across regimes
# - Random Forest (0.45): Best validation accuracy, good feature importance
# - XGBoost (0.20): Diversity, slightly different error patterns
#
# Total ML = 0.65 (< 0.70 limit)
# Heuristic = 0.35 (> 0.30 minimum)

ENSEMBLE_WEIGHTS = EnsembleWeights(
    heuristic_weight=0.35,
    primary_ml_weight=0.45,
    secondary_ml_weight=0.20,
    primary_ml_model="random_forest",
    secondary_ml_model="xgboost",
    heuristic_justification=(
        "Interpretability anchor. Stable baseline that performs well in "
        "novel regimes where ML may overfit to historical patterns."
    ),
    primary_ml_justification=(
        "Random Forest selected as primary ML model based on Phase 4.1 "
        "validation results: highest validation accuracy (0.78), good "
        "calibration, and interpretable feature importance via SHAP."
    ),
    secondary_ml_justification=(
        "XGBoost included for ensemble diversity. Slightly different "
        "error patterns than Random Forest, improving robustness."
    ),
)


def compute_weighted_ensemble(
    heuristic_score: float,
    ml_scores: dict[str, float],
    weights: EnsembleWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute weighted ensemble score from component scores.

    Handles missing ML models by renormalizing weights across available models.
    The heuristic weight remains unchanged; only the distribution within the
    ML weight block is adjusted.

    Args:
        heuristic_score: Heuristic risk score in [0, 1]
        ml_scores: Dict of model_name -> risk_score for ML models
        weights: Ensemble weight configuration (uses defaults if None)

    Returns:
        Tuple of (ensemble_score, contributions_dict)

    Raises:
        ValueError: If no valid ML model scores are provided
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    # Build configured ML weights dict
    configured_ml_weights: dict[str, float] = {
        weights.primary_ml_model: weights.primary_ml_weight,
    }
    if weights.secondary_ml_model and weights.secondary_ml_weight > 0:
        configured_ml_weights[weights.secondary_ml_model] = weights.secondary_ml_weight

    # Step 1: Filter ML weights to available models
    available_ml_weights = {
        model: weight
        for model, weight in configured_ml_weights.items()
        if model in ml_scores
    }

    # Step 2: Handle missing models safely
    if not available_ml_weights:
        raise ValueError(
            f"No valid ML model scores provided for ensemble. "
            f"Configured: {list(configured_ml_weights.keys())}, "
            f"Available: {list(ml_scores.keys())}"
        )

    # Step 3: Renormalize ML weights across available models
    # The total ML weight mass stays the same, only distribution changes
    total_configured_ml_weight = sum(available_ml_weights.values())
    total_ml_weight = weights.total_ml_weight  # Preserve original ML weight mass

    normalized_ml_weights = {
        model: (weight / total_configured_ml_weight) * total_ml_weight
        for model, weight in available_ml_weights.items()
    }

    # Initialize contributions
    contributions: dict[str, float] = {}

    # Heuristic contribution (unchanged)
    h_contrib = weights.heuristic_weight * heuristic_score
    contributions["heuristic"] = h_contrib

    # ML contributions using normalized weights
    ml_contrib_total = 0.0
    for model, norm_weight in normalized_ml_weights.items():
        score = ml_scores[model]
        contrib = norm_weight * score
        contributions[model] = contrib
        ml_contrib_total += contrib

    # Compute ensemble score
    ensemble_score = h_contrib + ml_contrib_total

    return ensemble_score, contributions


def compute_simple_average(
    heuristic_score: float,
    ml_scores: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Compute simple average of all available scores.

    Fallback method when specific models not available.

    Args:
        heuristic_score: Heuristic risk score
        ml_scores: Dict of ML model scores

    Returns:
        Tuple of (average_score, contributions_dict)
    """
    all_scores = {"heuristic": heuristic_score, **ml_scores}
    n_models = len(all_scores)
    weight = 1.0 / n_models

    contributions = {name: weight * score for name, score in all_scores.items()}
    ensemble_score = sum(contributions.values())

    return ensemble_score, contributions
