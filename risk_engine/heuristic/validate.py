"""
Validation for heuristic risk engine outputs.

CRITICAL RULES:
- Risk score must be in [0.0, 1.0]
- Sub-scores must be in [0.0, 1.0]
- All required fields must be present
- Feature names must match Phase 3 outputs
- No NaNs (use None)

Fail fast on violations.
"""

from __future__ import annotations

import math
from typing import Any

# Regime bands from spec
REGIME_BANDS: list[tuple[str, float, float]] = [
    ("LOW_RISK", 0.0, 0.25),
    ("MODERATE_RISK", 0.25, 0.50),
    ("HIGH_RISK", 0.50, 0.75),
    ("EXTREME_RISK", 0.75, 1.0),
]

# Required sub-scores
REQUIRED_SUB_SCORES = [
    "volatility_risk",
    "correlation_risk",
    "liquidity_risk",
    "sentiment_risk",
    "credit_risk",
]

# Valid Phase 3 feature names
PHASE_3_NUMERIC_FEATURES = [
    "realized_volatility_20d",
    "realized_volatility_60d",
    "volatility_ratio_20d_60d",
    "max_drawdown_20d",
    "max_drawdown_60d",
    "volume_zscore_20d",
    "volume_price_divergence",
    "vix_level",
]

PHASE_3_SENTIMENT_FEATURES = [
    "news_sentiment_daily",
    "news_sentiment_5d",
    "sentiment_volatility_20d",
]

PHASE_3_GRAPH_FEATURES = [
    "avg_pairwise_correlation_20d",
    "correlation_dispersion_20d",
    "sector_correlation_to_market",
    "network_centrality_change",
]

ALL_VALID_FEATURES = (
    PHASE_3_NUMERIC_FEATURES + PHASE_3_SENTIMENT_FEATURES + PHASE_3_GRAPH_FEATURES
)


def load_risk_spec() -> dict[str, Any]:
    """Load risk_score.yaml specification."""
    from importlib import resources

    import yaml

    try:
        with resources.files("core_specs").joinpath("risk_score.yaml").open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Risk score spec not found. Ensure core_specs package is installed."
        ) from None


def score_to_regime(score: float) -> str:
    """
    Map risk score to regime band.

    Args:
        score: Risk score in [0, 1]

    Returns:
        Regime name: LOW_RISK, MODERATE_RISK, HIGH_RISK, or EXTREME_RISK

    Bands (from spec):
        [0.00, 0.25) -> LOW_RISK
        [0.25, 0.50) -> MODERATE_RISK
        [0.50, 0.75) -> HIGH_RISK
        [0.75, 1.00] -> EXTREME_RISK
    """
    if score < 0.25:
        return "LOW_RISK"
    elif score < 0.50:
        return "MODERATE_RISK"
    elif score < 0.75:
        return "HIGH_RISK"
    else:
        return "EXTREME_RISK"


def validate_risk_score(score: float | None) -> tuple[bool, list[str]]:
    """
    Validate risk score value.

    Args:
        score: Risk score to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if score is None:
        # None is valid (indicates insufficient data)
        return True, []

    if math.isnan(score):
        errors.append("Risk score is NaN (use None for missing)")
        return False, errors

    if not (0.0 <= score <= 1.0):
        errors.append(f"Risk score {score} out of range [0, 1]")

    return len(errors) == 0, errors


def validate_sub_scores(
    sub_scores: dict[str, float | None],
) -> tuple[bool, list[str]]:
    """
    Validate sub-score values.

    Args:
        sub_scores: Dict of sub_score_name -> value

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required sub-scores exist
    for name in REQUIRED_SUB_SCORES:
        if name not in sub_scores:
            errors.append(f"Missing required sub-score: {name}")

    # Validate values
    for name, value in sub_scores.items():
        if value is None:
            continue  # None is valid

        if math.isnan(value):
            errors.append(f"Sub-score {name} is NaN (use None)")
            continue

        if not (0.0 <= value <= 1.0):
            errors.append(f"Sub-score {name} = {value} out of range [0, 1]")

    return len(errors) == 0, errors


def validate_feature_contributions(
    contributions: dict[str, float],
) -> tuple[bool, list[str]]:
    """
    Validate feature contribution dict.

    Args:
        contributions: Dict of feature_name -> contribution

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    for name, value in contributions.items():
        # Check feature name is valid
        if name not in ALL_VALID_FEATURES:
            errors.append(f"Unknown feature in contributions: {name}")

        # Check for NaN
        if math.isnan(value):
            errors.append(f"Feature contribution {name} is NaN")

    return len(errors) == 0, errors


def validate_input_features(
    numeric: dict[str, float | None] | None,
    sentiment: dict[str, float | None] | None,
    graph: dict[str, float | None] | None,
) -> tuple[bool, list[str]]:
    """
    Validate input feature dicts.

    Args:
        numeric: Numeric features dict
        sentiment: Sentiment features dict
        graph: Graph features dict

    Returns:
        Tuple of (is_valid, list of warnings)

    Note: Unknown features are warnings, not errors.
    """
    warnings = []

    # Check numeric features
    if numeric:
        for name in numeric:
            if name not in PHASE_3_NUMERIC_FEATURES:
                warnings.append(f"Unknown numeric feature: {name}")

    # Check sentiment features
    if sentiment:
        for name in sentiment:
            if name not in PHASE_3_SENTIMENT_FEATURES:
                warnings.append(f"Unknown sentiment feature: {name}")

    # Check graph features
    if graph:
        for name in graph:
            if name not in PHASE_3_GRAPH_FEATURES:
                warnings.append(f"Unknown graph feature: {name}")

    return True, warnings  # Warnings don't fail validation


def validate_risk_snapshot(
    risk_score: float | None,
    sub_scores: dict[str, float | None],
    feature_contributions: dict[str, float],
    regime: str,
) -> tuple[bool, list[str]]:
    """
    Validate complete risk score snapshot.

    Args:
        risk_score: Final risk score
        sub_scores: Sub-score dict
        feature_contributions: Feature contribution dict
        regime: Assigned regime

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Validate risk score
    valid, score_errors = validate_risk_score(risk_score)
    errors.extend(score_errors)

    # Validate sub-scores
    valid, sub_errors = validate_sub_scores(sub_scores)
    errors.extend(sub_errors)

    # Validate feature contributions
    valid, contrib_errors = validate_feature_contributions(feature_contributions)
    errors.extend(contrib_errors)

    # Validate regime
    valid_regimes = [band[0] for band in REGIME_BANDS]
    if regime not in valid_regimes:
        errors.append(f"Invalid regime: {regime}")

    # Check regime matches score
    if risk_score is not None:
        expected_regime = score_to_regime(risk_score)
        if regime != expected_regime:
            errors.append(
                f"Regime mismatch: score {risk_score} should be {expected_regime}, "
                f"got {regime}"
            )

    return len(errors) == 0, errors
