"""
Explicit weights for heuristic risk engine.

CRITICAL RULES:
- All weights are explicit constants (not learned)
- All weights are documented with rationale
- Weights are easy to change (single source of truth)
- No hidden logic

Weight structure:
1. Feature weights within each sub-score
2. Sub-score weights in final risk score

All weights are normalized to sum to 1.0 within their group.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubScoreDefinition:
    """
    Definition of a sub-score bucket.

    Attributes:
        name: Sub-score name (e.g., "volatility_risk")
        features: List of feature names contributing to this sub-score
        feature_weights: Dict of feature -> weight (normalized within sub-score)
        description: Human-readable description
        risk_interpretation: What high values mean
    """

    name: str
    features: list[str]
    feature_weights: dict[str, float]
    description: str
    risk_interpretation: str


# ==============================================================================
# SUB-SCORE DEFINITIONS
# ==============================================================================
# Each sub-score aggregates related features into a single interpretable bucket.

VOLATILITY_RISK = SubScoreDefinition(
    name="volatility_risk",
    features=[
        "realized_volatility_20d",
        "realized_volatility_60d",
        "volatility_ratio_20d_60d",
        "vix_level",
    ],
    feature_weights={
        "realized_volatility_20d": 0.30,  # Short-term vol most important
        "realized_volatility_60d": 0.15,  # Medium-term context
        "volatility_ratio_20d_60d": 0.20,  # Stress indicator
        "vix_level": 0.35,  # Market fear gauge
    },
    description="Risk from realized and implied volatility signals",
    risk_interpretation="High values indicate elevated market uncertainty and fear",
)

CORRELATION_RISK = SubScoreDefinition(
    name="correlation_risk",
    features=[
        "avg_pairwise_correlation_20d",
        "correlation_dispersion_20d",
        "sector_correlation_to_market",
        "network_centrality_change",
    ],
    feature_weights={
        "avg_pairwise_correlation_20d": 0.35,  # Primary correlation signal
        "correlation_dispersion_20d": 0.20,  # Uniformity of stress
        "sector_correlation_to_market": 0.25,  # Sector lockstep
        "network_centrality_change": 0.20,  # Network instability
    },
    description="Risk from cross-asset correlation dynamics",
    risk_interpretation="High values indicate diversification failure and systemic coupling",
)

LIQUIDITY_RISK = SubScoreDefinition(
    name="liquidity_risk",
    features=[
        "volume_zscore_20d",
        "volume_price_divergence",
    ],
    feature_weights={
        "volume_zscore_20d": 0.50,  # Volume abnormality
        "volume_price_divergence": 0.50,  # Price-volume stress
    },
    description="Risk from volume and market functioning indicators",
    risk_interpretation="High values indicate market stress and potential liquidity issues",
)

SENTIMENT_RISK = SubScoreDefinition(
    name="sentiment_risk",
    features=[
        "news_sentiment_daily",
        "news_sentiment_5d",
        "sentiment_volatility_20d",
    ],
    feature_weights={
        "news_sentiment_daily": 0.30,  # Current sentiment
        "news_sentiment_5d": 0.40,  # Trend sentiment (more stable)
        "sentiment_volatility_20d": 0.30,  # Sentiment instability
    },
    description="Risk from news sentiment deterioration",
    risk_interpretation="High values indicate negative market sentiment and fear",
)

CREDIT_RISK = SubScoreDefinition(
    name="credit_risk",
    features=[
        "max_drawdown_20d",
        "max_drawdown_60d",
    ],
    feature_weights={
        "max_drawdown_20d": 0.60,  # Recent drawdown
        "max_drawdown_60d": 0.40,  # Extended drawdown
    },
    description="Risk from drawdown and credit-related indicators",
    risk_interpretation="High values indicate recent losses and potential credit stress",
)

# All sub-score definitions
SUB_SCORE_DEFINITIONS: dict[str, SubScoreDefinition] = {
    "volatility_risk": VOLATILITY_RISK,
    "correlation_risk": CORRELATION_RISK,
    "liquidity_risk": LIQUIDITY_RISK,
    "sentiment_risk": SENTIMENT_RISK,
    "credit_risk": CREDIT_RISK,
}


# ==============================================================================
# FINAL RISK SCORE WEIGHTS
# ==============================================================================
# Weights for combining sub-scores into final risk score.

FINAL_SCORE_WEIGHTS: dict[str, float] = {
    "volatility_risk": 0.30,  # Primary stress indicator
    "correlation_risk": 0.25,  # Systemic coupling
    "liquidity_risk": 0.15,  # Market functioning
    "sentiment_risk": 0.15,  # Behavioral component
    "credit_risk": 0.15,  # Loss-based component
}

# Verify weights sum to 1.0
_total_weight = sum(FINAL_SCORE_WEIGHTS.values())
assert abs(_total_weight - 1.0) < 1e-10, f"Final weights must sum to 1.0, got {_total_weight}"


def get_sub_score_definition(name: str) -> SubScoreDefinition | None:
    """Get definition for a sub-score by name."""
    return SUB_SCORE_DEFINITIONS.get(name)


def get_sub_score_names() -> list[str]:
    """Get list of all sub-score names."""
    return list(SUB_SCORE_DEFINITIONS.keys())


def get_final_weight(sub_score: str) -> float:
    """Get weight for a sub-score in final risk score."""
    return FINAL_SCORE_WEIGHTS.get(sub_score, 0.0)


def get_feature_weight(sub_score: str, feature: str) -> float:
    """Get weight for a feature within a sub-score."""
    defn = SUB_SCORE_DEFINITIONS.get(sub_score)
    if defn is None:
        return 0.0
    return defn.feature_weights.get(feature, 0.0)


def get_all_weights_summary() -> dict[str, dict[str, float]]:
    """
    Get complete weights summary for documentation.

    Returns:
        Dict with structure:
        {
            "sub_score_weights": {"volatility_risk": 0.30, ...},
            "feature_weights": {
                "volatility_risk": {"realized_volatility_20d": 0.30, ...},
                ...
            }
        }
    """
    return {
        "sub_score_weights": FINAL_SCORE_WEIGHTS.copy(),
        "feature_weights": {
            name: defn.feature_weights.copy()
            for name, defn in SUB_SCORE_DEFINITIONS.items()
        },
    }
