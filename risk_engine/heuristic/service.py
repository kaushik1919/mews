"""
Heuristic Risk Engine Public API.

MEWS-FIN Phase 4.0: Combine multimodal feature snapshots into
a single interpretable risk score using explicit rules and weights.

NOT a prediction model. NOT a trading signal.
This is a baseline sanity check, explainable to a risk committee.

Usage:
    from risk_engine.heuristic import compute_risk_score

    snapshot = compute_risk_score(
        numeric_features={
            "realized_volatility_20d": 0.25,
            "max_drawdown_20d": 0.15,
            "vix_level": 22.5,
            ...
        },
        sentiment_features={
            "news_sentiment_daily": -0.3,
            ...
        },
        graph_features={
            "avg_pairwise_correlation_20d": 0.65,
            ...
        },
        as_of="2024-01-15T16:00:00Z",
    )

    print(snapshot.risk_score)  # 0.58
    print(snapshot.regime)      # "HIGH_RISK"
    print(snapshot.regime_rationale)  # "Elevated risk driven by..."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .explain import compute_explainability_report
from .normalization import normalize_features
from .subscores import (
    compute_all_sub_scores,
    compute_final_risk_score,
)
from .validate import (
    score_to_regime,
    validate_input_features,
    validate_risk_snapshot,
)
from .weights import FINAL_SCORE_WEIGHTS, SUB_SCORE_DEFINITIONS


@dataclass
class RiskScoreSnapshot:
    """
    Complete risk score output with full explainability.

    All fields follow core-specs/risk_score.yaml contract.

    Attributes:
        risk_score: Final risk score in [0, 1]. None if insufficient data.
        regime: Risk regime: LOW_RISK, MODERATE_RISK, HIGH_RISK, EXTREME_RISK.
        as_of: Timestamp of the snapshot (ISO 8601).
        sub_scores: Dict of sub_score_name -> value.
        feature_contributions: Dict of feature_name -> contribution to final score.
        sub_score_contributions: Dict of sub_score_name -> contribution to final score.
        dominant_factors: Top features driving the risk score.
        regime_rationale: Human-readable explanation of the regime.
        warnings: List of validation warnings.
        version: Engine version for reproducibility.
    """

    risk_score: float | None
    regime: str
    as_of: str
    sub_scores: dict[str, float | None]
    feature_contributions: dict[str, float]
    sub_score_contributions: dict[str, float]
    dominant_factors: list[str]
    regime_rationale: str
    warnings: list[str] = field(default_factory=list)
    version: str = "heuristic-v1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "risk_score": self.risk_score,
            "regime": self.regime,
            "as_of": self.as_of,
            "sub_scores": self.sub_scores,
            "feature_contributions": self.feature_contributions,
            "sub_score_contributions": self.sub_score_contributions,
            "dominant_factors": self.dominant_factors,
            "regime_rationale": self.regime_rationale,
            "warnings": self.warnings,
            "version": self.version,
        }


def compute_risk_score(
    numeric_features: dict[str, float | None] | None = None,
    sentiment_features: dict[str, float | None] | None = None,
    graph_features: dict[str, float | None] | None = None,
    as_of: str | datetime | None = None,
    fail_on_validation_error: bool = True,
) -> RiskScoreSnapshot:
    """
    Compute risk score from multimodal feature snapshots.

    This is the main entry point for the heuristic risk engine.

    Args:
        numeric_features: Dict of numeric feature values (Phase 3).
        sentiment_features: Dict of sentiment feature values (Phase 3).
        graph_features: Dict of graph feature values (Phase 3).
        as_of: Timestamp of the snapshot. Defaults to now.
        fail_on_validation_error: If True, raise on validation errors.

    Returns:
        RiskScoreSnapshot with full explainability.

    Raises:
        ValueError: If fail_on_validation_error and validation fails.

    Example:
        >>> snapshot = compute_risk_score(
        ...     numeric_features={"realized_volatility_20d": 0.25},
        ...     as_of="2024-01-15T16:00:00Z",
        ... )
        >>> snapshot.risk_score
        0.42
    """
    # Handle timestamp
    if as_of is None:
        as_of_str = datetime.utcnow().isoformat() + "Z"
    elif isinstance(as_of, datetime):
        as_of_str = as_of.isoformat() + "Z"
    else:
        as_of_str = as_of

    # Validate inputs (warnings only)
    _, input_warnings = validate_input_features(
        numeric_features, sentiment_features, graph_features
    )

    # Merge all features into single dict
    raw_features: dict[str, float | None] = {}
    if numeric_features:
        raw_features.update(numeric_features)
    if sentiment_features:
        raw_features.update(sentiment_features)
    if graph_features:
        raw_features.update(graph_features)

    # Handle empty input
    if not raw_features:
        return RiskScoreSnapshot(
            risk_score=None,
            regime="LOW_RISK",  # Default when no data
            as_of=as_of_str,
            sub_scores=dict.fromkeys(FINAL_SCORE_WEIGHTS),
            feature_contributions={},
            sub_score_contributions={},
            dominant_factors=[],
            regime_rationale="Insufficient data to compute risk score.",
            warnings=["No features provided"],
        )

    # Step 1: Normalize features to [0, 1]
    normalized = normalize_features(raw_features)

    # Step 2: Compute sub-scores
    sub_score_results = compute_all_sub_scores(normalized)

    # Step 3: Compute final risk score
    final_score, sub_score_contribs = compute_final_risk_score(sub_score_results)

    # Step 4: Determine regime
    if final_score is not None:
        regime = score_to_regime(final_score)
    else:
        regime = "LOW_RISK"  # Default when score is None

    # Step 5: Compute explainability
    explainability = compute_explainability_report(
        risk_score=final_score,
        regime=regime,
        sub_score_results=sub_score_results,
    )

    # Build sub_scores dict
    sub_scores_dict: dict[str, float | None] = {}
    for name, result in sub_score_results.items():
        sub_scores_dict[name] = result.value

    # Validate output
    is_valid, validation_errors = validate_risk_snapshot(
        risk_score=final_score,
        sub_scores=sub_scores_dict,
        feature_contributions=explainability.feature_contributions,
        regime=regime,
    )

    if not is_valid and fail_on_validation_error:
        raise ValueError(f"Risk score validation failed: {validation_errors}")

    # Combine warnings
    all_warnings = input_warnings + validation_errors

    return RiskScoreSnapshot(
        risk_score=final_score,
        regime=regime,
        as_of=as_of_str,
        sub_scores=sub_scores_dict,
        feature_contributions=explainability.feature_contributions,
        sub_score_contributions=explainability.sub_score_contributions,
        dominant_factors=explainability.dominant_factors,
        regime_rationale=explainability.regime_rationale,
        warnings=all_warnings,
    )


def get_weight_config() -> dict[str, Any]:
    """
    Return the current weight configuration for transparency.

    Useful for auditing and documentation.

    Returns:
        Dict with sub-score definitions and final weights.
    """
    sub_scores = {}
    for name, definition in SUB_SCORE_DEFINITIONS.items():
        sub_scores[name] = {
            "features": definition.features,
            "feature_weights": definition.feature_weights,
            "description": definition.description,
            "risk_interpretation": definition.risk_interpretation,
        }

    return {
        "sub_scores": sub_scores,
        "final_score_weights": FINAL_SCORE_WEIGHTS,
        "version": "heuristic-v1.0",
    }


def get_historical_calibration_info() -> dict[str, Any]:
    """
    Return historical calibration anchors for validation.

    From core-specs/risk_score.yaml.

    Returns:
        Dict of calibration periods with expected score ranges.
    """
    return {
        "anchors": [
            {
                "period": "2008 Global Financial Crisis",
                "expected_range": [0.85, 1.00],
                "notes": "Peak systemic stress, extreme regime",
            },
            {
                "period": "2020 COVID-19 March",
                "expected_range": [0.80, 0.95],
                "notes": "Liquidity crisis, extreme volatility",
            },
            {
                "period": "2017 Low Volatility",
                "expected_range": [0.05, 0.20],
                "notes": "Exceptionally calm markets, low risk",
            },
            {
                "period": "2022 Rate Hiking Cycle",
                "expected_range": [0.45, 0.65],
                "notes": "Elevated but not extreme stress",
            },
        ],
        "notes": (
            "These anchors are for manual validation only. "
            "Heuristic engine uses explicit rules, not calibration."
        ),
    }
