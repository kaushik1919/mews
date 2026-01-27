"""
Feature normalization for heuristic risk engine.

CRITICAL RULES:
- Normalize features individually
- Use robust rules (z-score with clipping or min-max with caps)
- No global statistics (each feature normalized independently)
- No future data (all params are constants, not learned)
- All outputs in [0, 1] range

Normalization is explicit and documented for each feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationParams:
    """
    Parameters for feature normalization.

    Attributes:
        method: 'zscore' or 'minmax'
        center: Mean or min for normalization
        scale: Std dev or range for normalization
        clip_low: Minimum normalized value (before [0,1] mapping)
        clip_high: Maximum normalized value (before [0,1] mapping)
        invert: If True, higher raw values → lower normalized values
    """

    method: str
    center: float
    scale: float
    clip_low: float = -3.0
    clip_high: float = 3.0
    invert: bool = False


# ==============================================================================
# NORMALIZATION PARAMETERS BY FEATURE
# ==============================================================================
# These are based on historical market behavior. All parameters are explicit
# constants, not learned from data. They represent domain knowledge about
# typical ranges and stress levels for each feature.

FEATURE_NORMALIZATION: dict[str, NormalizationParams] = {
    # -------------------------------------------------------------------------
    # NUMERIC FEATURES (Phase 3.0)
    # -------------------------------------------------------------------------
    # Volatility features: Higher = more risk
    "realized_volatility_20d": NormalizationParams(
        method="zscore",
        center=0.15,  # Typical annualized vol ~15%
        scale=0.10,  # Std dev of vol
        clip_low=-2.0,
        clip_high=4.0,  # Extreme vol can be 4+ std devs
        invert=False,
    ),
    "realized_volatility_60d": NormalizationParams(
        method="zscore",
        center=0.15,
        scale=0.08,  # Less variable than 20d
        clip_low=-2.0,
        clip_high=4.0,
        invert=False,
    ),
    "volatility_ratio_20d_60d": NormalizationParams(
        method="zscore",
        center=1.0,  # Ratio of 1 = equal vol
        scale=0.3,  # Typical variation
        clip_low=-2.0,
        clip_high=3.0,  # >1 means short-term stress
        invert=False,
    ),
    # Drawdown features: More negative = more risk
    "max_drawdown_20d": NormalizationParams(
        method="minmax",
        center=0.0,  # 0% drawdown = no stress
        scale=-0.20,  # -20% drawdown = high stress
        clip_low=0.0,
        clip_high=1.0,
        invert=False,  # Already scaled so higher raw = more negative = more risk
    ),
    "max_drawdown_60d": NormalizationParams(
        method="minmax",
        center=0.0,
        scale=-0.30,  # -30% drawdown = high stress
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # Volume features: Z-score already, higher = more unusual
    "volume_zscore_20d": NormalizationParams(
        method="zscore",
        center=0.0,
        scale=2.0,  # Normalize the z-score itself
        clip_low=-2.0,
        clip_high=4.0,  # High volume spikes
        invert=False,
    ),
    # Divergence: Higher = stress
    "volume_price_divergence": NormalizationParams(
        method="zscore",
        center=0.0,
        scale=0.5,
        clip_low=-2.0,
        clip_high=3.0,
        invert=False,
    ),
    # VIX: Explicit levels
    "vix_level": NormalizationParams(
        method="minmax",
        center=12.0,  # Low VIX baseline
        scale=50.0,  # VIX 50+ = extreme
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # -------------------------------------------------------------------------
    # SENTIMENT FEATURES (Phase 3.1)
    # -------------------------------------------------------------------------
    # Sentiment: More negative = more risk
    "news_sentiment_daily": NormalizationParams(
        method="minmax",
        center=1.0,  # +1 = positive
        scale=-2.0,  # -1 = negative, range is 2
        clip_low=0.0,
        clip_high=1.0,
        invert=False,  # After transform: 1=neg, 0=pos
    ),
    "news_sentiment_5d": NormalizationParams(
        method="minmax",
        center=1.0,
        scale=-2.0,
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # Sentiment volatility: Higher = more risk
    "sentiment_volatility_20d": NormalizationParams(
        method="zscore",
        center=0.3,  # Typical sentiment std
        scale=0.2,
        clip_low=-2.0,
        clip_high=3.0,
        invert=False,
    ),
    # -------------------------------------------------------------------------
    # GRAPH FEATURES (Phase 3.2)
    # -------------------------------------------------------------------------
    # Correlation: Higher = more risk (diversification failure)
    "avg_pairwise_correlation_20d": NormalizationParams(
        method="minmax",
        center=0.2,  # Low correlation baseline
        scale=0.6,  # Range to high correlation
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # Dispersion: Lower = more risk (uniform stress)
    "correlation_dispersion_20d": NormalizationParams(
        method="minmax",
        center=0.3,  # Normal dispersion
        scale=-0.25,  # Low dispersion = stress
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # Sector correlation: Higher = more risk
    "sector_correlation_to_market": NormalizationParams(
        method="minmax",
        center=0.5,  # Normal sector-market correlation
        scale=0.4,  # Range to high
        clip_low=0.0,
        clip_high=1.0,
        invert=False,
    ),
    # Centrality change: Higher = more risk (instability)
    "network_centrality_change": NormalizationParams(
        method="zscore",
        center=0.05,  # Typical change
        scale=0.05,  # Std dev of change
        clip_low=-2.0,
        clip_high=4.0,
        invert=False,
    ),
}


def normalize_feature(
    name: str,
    value: float | None,
) -> float | None:
    """
    Normalize a single feature value to [0, 1] range.

    Args:
        name: Feature name (must be in FEATURE_NORMALIZATION)
        value: Raw feature value, or None

    Returns:
        Normalized value in [0, 1], or None if input is None

    Raises:
        ValueError: If feature name unknown

    Mathematical definition:
        For z-score method:
            z = (value - center) / scale
            z_clipped = clip(z, clip_low, clip_high)
            normalized = (z_clipped - clip_low) / (clip_high - clip_low)

        For minmax method:
            raw_norm = (value - center) / scale
            normalized = clip(raw_norm, 0, 1)

        If invert: normalized = 1 - normalized
    """
    if value is None:
        return None

    if math.isnan(value):
        return None

    if name not in FEATURE_NORMALIZATION:
        raise ValueError(f"Unknown feature for normalization: {name}")

    params = FEATURE_NORMALIZATION[name]

    if params.method == "zscore":
        # Z-score normalization
        z = (value - params.center) / params.scale
        z_clipped = max(params.clip_low, min(params.clip_high, z))
        # Map clipped z-score to [0, 1]
        normalized = (z_clipped - params.clip_low) / (
            params.clip_high - params.clip_low
        )
    elif params.method == "minmax":
        # Min-max normalization
        raw_norm = (value - params.center) / params.scale
        normalized = max(0.0, min(1.0, raw_norm))
    else:
        raise ValueError(f"Unknown normalization method: {params.method}")

    if params.invert:
        normalized = 1.0 - normalized

    return normalized


def normalize_features(
    features: dict[str, float | None],
) -> dict[str, float | None]:
    """
    Normalize all features in a dict.

    Args:
        features: Dict of feature_name -> raw value

    Returns:
        Dict of feature_name -> normalized value in [0, 1]

    Features not in FEATURE_NORMALIZATION are passed through as None.
    """
    result: dict[str, float | None] = {}
    for name, value in features.items():
        if name in FEATURE_NORMALIZATION:
            result[name] = normalize_feature(name, value)
        else:
            # Unknown feature - pass as None
            result[name] = None
    return result


def get_normalization_params(name: str) -> NormalizationParams | None:
    """Get normalization parameters for a feature."""
    return FEATURE_NORMALIZATION.get(name)


def list_normalized_features() -> list[str]:
    """List all features with normalization parameters."""
    return list(FEATURE_NORMALIZATION.keys())
