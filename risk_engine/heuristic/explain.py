"""
Explainability logic for heuristic risk engine.

CRITICAL RULES:
- Compute feature-level contributions
- Compute sub-score contributions
- Identify dominant drivers (top N)
- Generate human-readable regime rationale

This is the explainability reference for Phase 4.1+ ML models.
Any future model must match this level of transparency.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.heuristic.subscores import SubScoreResult
from risk_engine.heuristic.weights import (
    FINAL_SCORE_WEIGHTS,
)


@dataclass
class ExplainabilityReport:
    """
    Complete explainability report for a risk score.

    Attributes:
        feature_contributions: Dict of feature -> contribution to final score
        sub_score_contributions: Dict of sub_score -> contribution to final score
        dominant_factors: List of top features driving the score
        regime_rationale: Human-readable explanation of regime
    """

    feature_contributions: dict[str, float]
    sub_score_contributions: dict[str, float]
    dominant_factors: list[str]
    regime_rationale: str


def compute_feature_contributions(
    sub_score_results: dict[str, SubScoreResult],
) -> dict[str, float]:
    """
    Compute contribution of each feature to the final risk score.

    Each feature's contribution is:
        contribution = (normalized_value * feature_weight_in_subscore)
                       * (subscore_weight_in_final)

    Features with None values have 0 contribution.

    Args:
        sub_score_results: Dict of sub_score_name -> SubScoreResult

    Returns:
        Dict of feature_name -> contribution to final score
    """
    contributions: dict[str, float] = {}

    # Calculate total weight of valid sub-scores for redistribution
    total_sub_weight = sum(
        FINAL_SCORE_WEIGHTS.get(name, 0.0)
        for name, result in sub_score_results.items()
        if result.value is not None
    )

    if total_sub_weight == 0:
        return contributions

    for sub_name, sub_result in sub_score_results.items():
        if sub_result.value is None:
            continue

        # Get effective sub-score weight (redistributed)
        sub_weight = FINAL_SCORE_WEIGHTS.get(sub_name, 0.0) / total_sub_weight

        # Each feature's contribution in the sub-score is already weighted
        # within the sub-score. Now scale by sub-score's weight in final.
        for feature, feature_contrib in sub_result.feature_contributions.items():
            # feature_contrib is already the weighted contribution within sub-score
            # Multiply by sub-score's effective weight in final score
            final_contrib = feature_contrib * sub_weight
            contributions[feature] = final_contrib

    return contributions


def compute_sub_score_contributions(
    sub_score_results: dict[str, SubScoreResult],
) -> dict[str, float]:
    """
    Compute contribution of each sub-score to the final risk score.

    Args:
        sub_score_results: Dict of sub_score_name -> SubScoreResult

    Returns:
        Dict of sub_score_name -> contribution to final score
    """
    contributions: dict[str, float] = {}

    # Calculate total weight of valid sub-scores
    total_weight = sum(
        FINAL_SCORE_WEIGHTS.get(name, 0.0)
        for name, result in sub_score_results.items()
        if result.value is not None
    )

    if total_weight == 0:
        return contributions

    for name, result in sub_score_results.items():
        if result.value is None:
            contributions[name] = 0.0
            continue

        effective_weight = FINAL_SCORE_WEIGHTS.get(name, 0.0) / total_weight
        contributions[name] = result.value * effective_weight

    return contributions


def identify_dominant_factors(
    feature_contributions: dict[str, float],
    top_n: int = 3,
) -> list[str]:
    """
    Identify top N features driving the risk score.

    Args:
        feature_contributions: Dict of feature -> contribution
        top_n: Number of top factors to return

    Returns:
        List of feature names, sorted by contribution (descending)
    """
    if not feature_contributions:
        return []

    # Sort by absolute contribution (descending)
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return [name for name, _ in sorted_features[:top_n]]


def generate_regime_rationale(
    regime: str,
    sub_score_results: dict[str, SubScoreResult],
    dominant_factors: list[str],
) -> str:
    """
    Generate human-readable explanation of regime assignment.

    Args:
        regime: Current regime name (e.g., "HIGH_RISK")
        sub_score_results: Sub-score computation results
        dominant_factors: Top features driving the score

    Returns:
        Human-readable rationale string
    """
    # Identify elevated sub-scores (> 0.5)
    elevated_subs = []
    for name, result in sub_score_results.items():
        if result.value is not None and result.value > 0.5:
            elevated_subs.append(name.replace("_", " ").title())

    # Build rationale
    if regime == "EXTREME_RISK":
        if elevated_subs:
            return f"Crisis-level stress across {', '.join(elevated_subs[:3])}"
        return "Multiple stress indicators at extreme levels"

    elif regime == "HIGH_RISK":
        if elevated_subs:
            return f"Elevated {' and '.join(elevated_subs[:2])}"
        if dominant_factors:
            factor_names = [f.replace("_", " ") for f in dominant_factors[:2]]
            return f"Elevated {' and '.join(factor_names)}"
        return "Multiple stress indicators significantly elevated"

    elif regime == "MODERATE_RISK":
        if elevated_subs:
            return f"Some elevation in {elevated_subs[0]}"
        return "Some stress indicators elevated but not at crisis levels"

    else:  # LOW_RISK
        return "Key stress indicators within historical norms"


def compute_explainability_report(
    risk_score: float | None,
    regime: str,
    sub_score_results: dict[str, SubScoreResult],
) -> ExplainabilityReport:
    """
    Compute complete explainability report.

    Args:
        risk_score: Final risk score (or None)
        regime: Assigned regime
        sub_score_results: Sub-score computation results

    Returns:
        ExplainabilityReport with all explanations
    """
    feature_contribs = compute_feature_contributions(sub_score_results)
    sub_score_contribs = compute_sub_score_contributions(sub_score_results)
    dominant = identify_dominant_factors(feature_contribs, top_n=3)
    rationale = generate_regime_rationale(regime, sub_score_results, dominant)

    return ExplainabilityReport(
        feature_contributions=feature_contribs,
        sub_score_contributions=sub_score_contribs,
        dominant_factors=dominant,
        regime_rationale=rationale,
    )
