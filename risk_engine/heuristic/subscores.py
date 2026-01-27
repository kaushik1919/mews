"""
Sub-score computation for heuristic risk engine.

CRITICAL RULES:
- Aggregate normalized features into interpretable buckets
- Each sub-score is in [0, 1] range
- Handle missing inputs gracefully (skip with weight redistribution)
- No hidden logic

Sub-scores:
- volatility_risk: Realized and implied volatility
- correlation_risk: Cross-asset correlation dynamics
- liquidity_risk: Volume and market functioning
- sentiment_risk: News sentiment deterioration
- credit_risk: Drawdown and credit-related indicators
"""

from __future__ import annotations

from dataclasses import dataclass

from risk_engine.heuristic.normalization import normalize_feature
from risk_engine.heuristic.weights import (
    FINAL_SCORE_WEIGHTS,
    SUB_SCORE_DEFINITIONS,
    SubScoreDefinition,
)


@dataclass
class SubScoreResult:
    """
    Result of computing a single sub-score.

    Attributes:
        name: Sub-score name
        value: Computed sub-score in [0, 1], or None if insufficient data
        feature_contributions: Dict of feature -> contribution to this sub-score
        features_used: Number of features with valid values
        features_total: Total number of features in definition
    """

    name: str
    value: float | None
    feature_contributions: dict[str, float]
    features_used: int
    features_total: int


def compute_sub_score(
    definition: SubScoreDefinition,
    normalized_features: dict[str, float | None],
    min_features_required: int = 1,
) -> SubScoreResult:
    """
    Compute a single sub-score from normalized features.

    Args:
        definition: Sub-score definition with features and weights
        normalized_features: Dict of feature -> normalized value [0,1]
        min_features_required: Minimum features needed to compute score

    Returns:
        SubScoreResult with value and contributions

    Algorithm:
        1. Filter to features with valid (non-None) values
        2. If fewer than min_features_required, return None
        3. Redistribute weights among valid features
        4. Compute weighted sum
        5. Track individual feature contributions

    Missing data handling:
        Features with None values are excluded, and their weights
        are redistributed proportionally to available features.
    """
    feature_contributions: dict[str, float] = {}
    features_used = 0
    total_weight = 0.0
    weighted_sum = 0.0

    # First pass: identify valid features and total available weight
    valid_features: dict[str, float] = {}
    for feature in definition.features:
        norm_value = normalized_features.get(feature)
        if norm_value is not None:
            valid_features[feature] = norm_value
            total_weight += definition.feature_weights[feature]

    features_used = len(valid_features)

    # Check minimum requirement
    if features_used < min_features_required or total_weight == 0:
        return SubScoreResult(
            name=definition.name,
            value=None,
            feature_contributions={},
            features_used=0,
            features_total=len(definition.features),
        )

    # Second pass: compute weighted sum with redistributed weights
    for feature, norm_value in valid_features.items():
        original_weight = definition.feature_weights[feature]
        # Redistribute weight proportionally
        effective_weight = original_weight / total_weight
        contribution = norm_value * effective_weight
        weighted_sum += contribution
        feature_contributions[feature] = contribution

    # Clamp to [0, 1]
    sub_score_value = max(0.0, min(1.0, weighted_sum))

    return SubScoreResult(
        name=definition.name,
        value=sub_score_value,
        feature_contributions=feature_contributions,
        features_used=features_used,
        features_total=len(definition.features),
    )


def compute_all_sub_scores(
    normalized_features: dict[str, float | None],
) -> dict[str, SubScoreResult]:
    """
    Compute all sub-scores from normalized features.

    Args:
        normalized_features: Dict of feature -> normalized value

    Returns:
        Dict of sub_score_name -> SubScoreResult
    """
    results: dict[str, SubScoreResult] = {}
    for name, definition in SUB_SCORE_DEFINITIONS.items():
        results[name] = compute_sub_score(definition, normalized_features)
    return results


def compute_final_risk_score(
    sub_scores: dict[str, SubScoreResult],
    min_sub_scores_required: int = 3,
) -> tuple[float | None, dict[str, float]]:
    """
    Compute final risk score from sub-scores.

    Args:
        sub_scores: Dict of sub_score_name -> SubScoreResult
        min_sub_scores_required: Minimum sub-scores needed

    Returns:
        Tuple of (risk_score, sub_score_contributions)
        risk_score is None if insufficient sub-scores

    Algorithm:
        1. Filter to sub-scores with valid (non-None) values
        2. Redistribute weights among valid sub-scores
        3. Compute weighted sum
        4. Track sub-score contributions
    """
    sub_score_contributions: dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    # Identify valid sub-scores
    valid_sub_scores: dict[str, float] = {}
    for name, result in sub_scores.items():
        if result.value is not None:
            valid_sub_scores[name] = result.value
            total_weight += FINAL_SCORE_WEIGHTS.get(name, 0.0)

    # Check minimum requirement
    if len(valid_sub_scores) < min_sub_scores_required or total_weight == 0:
        return None, {}

    # Compute weighted sum with redistributed weights
    for name, value in valid_sub_scores.items():
        original_weight = FINAL_SCORE_WEIGHTS.get(name, 0.0)
        effective_weight = original_weight / total_weight
        contribution = value * effective_weight
        weighted_sum += contribution
        sub_score_contributions[name] = contribution

    # Clamp to [0, 1]
    risk_score = max(0.0, min(1.0, weighted_sum))

    return risk_score, sub_score_contributions


def compute_from_raw_features(
    numeric_features: dict[str, float | None],
    sentiment_features: dict[str, float | None],
    graph_features: dict[str, float | None],
) -> tuple[float | None, dict[str, float | None], dict[str, SubScoreResult]]:
    """
    Compute risk score from raw (unnormalized) features.

    This is a convenience function that:
    1. Combines all feature dicts
    2. Normalizes all features
    3. Computes sub-scores
    4. Computes final risk score

    Args:
        numeric_features: Phase 3.0 features
        sentiment_features: Phase 3.1 features
        graph_features: Phase 3.2 features

    Returns:
        Tuple of:
        - risk_score (or None)
        - sub_scores dict (name -> value or None)
        - full sub_score_results dict
    """
    # Combine all features
    all_features: dict[str, float | None] = {}
    all_features.update(numeric_features or {})
    all_features.update(sentiment_features or {})
    all_features.update(graph_features or {})

    # Normalize
    normalized: dict[str, float | None] = {}
    for name, value in all_features.items():
        normalized[name] = normalize_feature(name, value)

    # Compute sub-scores
    sub_score_results = compute_all_sub_scores(normalized)

    # Compute final score
    risk_score, _ = compute_final_risk_score(sub_score_results)

    # Extract simple sub-score values
    sub_score_values: dict[str, float | None] = {
        name: result.value for name, result in sub_score_results.items()
    }

    return risk_score, sub_score_values, sub_score_results
