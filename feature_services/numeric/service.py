"""
Numeric Feature Service - Public API.

This is the main entry point for computing numeric systemic risk features.
The service is STATELESS - no caching, no persistence, pure functions.

Usage:
    from feature_services.numeric import compute_numeric_features

    snapshot = compute_numeric_features(
        datasets={"market_prices": df_prices, "volatility_indices": df_vix},
        as_of=pd.Timestamp("2024-01-15 21:00:00", tz="UTC"),
    )

Output format:
    {
        "timestamp": "2024-01-15T21:00:00+00:00",
        "features": {
            "realized_volatility_20d": 0.156,
            "max_drawdown_60d": -0.18,
            ...
        }
    }
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .drawdown import compute_max_drawdown_20d, compute_max_drawdown_60d
from .liquidity import compute_volume_price_divergence, compute_volume_zscore_20d
from .validate import get_phase_30_feature_names, validate_feature_snapshot, validate_input_datasets
from .volatility import (
    compute_realized_volatility_20d,
    compute_realized_volatility_60d,
    compute_vix_level,
    compute_volatility_ratio,
)


@dataclass
class NumericFeatureSnapshot:
    """
    A point-in-time snapshot of numeric features.

    Attributes:
        timestamp: The as_of timestamp (UTC)
        features: Dict of feature_name -> value (or None if missing data)
        asset_id: Asset the features were computed for (default: SPY)
        metadata: Optional computation metadata
    """

    timestamp: pd.Timestamp
    features: dict[str, float | None]
    asset_id: str = "SPY"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "features": self.features,
            "metadata": self.metadata,
        }

    @property
    def is_complete(self) -> bool:
        """Check if all features have non-None values."""
        return all(v is not None for v in self.features.values())

    @property
    def missing_features(self) -> list[str]:
        """Get list of features with None values."""
        return [k for k, v in self.features.items() if v is None]


def compute_numeric_features(
    datasets: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
    vix_index_id: str = "^VIX",
    validate: bool = True,
) -> NumericFeatureSnapshot:
    """
    Compute all Phase 3.0 numeric features as of a specific timestamp.

    This function is STATELESS and PURE:
    - No caching or persistence
    - No side effects
    - Deterministic: same inputs -> same outputs
    - Only uses data with timestamp <= as_of (no lookahead)

    Args:
        datasets: Dict of dataset_name -> DataFrame
            Required: "market_prices", "volatility_indices"
            DataFrames must have Phase 2 output schema
        as_of: Reference timestamp (must be UTC-aware)
            All computations use only data <= as_of
        asset_id: Asset to compute features for (default: SPY)
        vix_index_id: VIX index identifier (default: ^VIX)
        validate: Whether to validate inputs and outputs (default: True)

    Returns:
        NumericFeatureSnapshot with all Phase 3.0 features
        Missing data -> value = None

    Raises:
        ValueError: If validation fails and validate=True

    Feature list (Phase 3.0):
        - realized_volatility_20d: Annualized 20-day volatility
        - realized_volatility_60d: Annualized 60-day volatility
        - volatility_ratio_20d_60d: Short/medium term volatility ratio
        - max_drawdown_20d: 20-day maximum peak-to-trough decline
        - max_drawdown_60d: 60-day maximum peak-to-trough decline
        - volume_zscore_20d: Volume z-score vs trailing distribution
        - volume_price_divergence: Return-volume correlation
        - vix_level: CBOE VIX index level
    """
    # Ensure as_of is timezone-aware
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (UTC)")

    # Validate inputs
    if validate:
        is_valid, errors = validate_input_datasets(datasets)
        if not is_valid:
            raise ValueError(f"Invalid input datasets: {errors}")

    # Extract datasets (with empty DataFrame fallback for missing)
    market_prices = datasets.get("market_prices", pd.DataFrame())
    volatility_indices = datasets.get("volatility_indices", pd.DataFrame())

    # Compute features
    features: dict[str, float | None] = {}

    # Volatility features
    vol_20d_raw = compute_realized_volatility_20d(
        market_prices, as_of, asset_id=asset_id, normalize=False
    )
    vol_60d_raw = compute_realized_volatility_60d(
        market_prices, as_of, asset_id=asset_id, normalize=False
    )

    features["realized_volatility_20d"] = vol_20d_raw
    features["realized_volatility_60d"] = vol_60d_raw
    features["volatility_ratio_20d_60d"] = compute_volatility_ratio(
        vol_20d_raw, vol_60d_raw
    )

    # Drawdown features
    features["max_drawdown_20d"] = compute_max_drawdown_20d(
        market_prices, as_of, asset_id=asset_id
    )
    features["max_drawdown_60d"] = compute_max_drawdown_60d(
        market_prices, as_of, asset_id=asset_id
    )

    # Liquidity features
    features["volume_zscore_20d"] = compute_volume_zscore_20d(
        market_prices, as_of, asset_id=asset_id
    )
    features["volume_price_divergence"] = compute_volume_price_divergence(
        market_prices, as_of, asset_id=asset_id
    )

    # VIX level (from volatility_indices)
    features["vix_level"] = compute_vix_level(
        volatility_indices, as_of, index_id=vix_index_id, normalize=False
    )

    # Validate output
    if validate:
        is_valid, errors = validate_feature_snapshot(features, strict=True)
        if not is_valid:
            raise ValueError(f"Invalid feature snapshot: {errors}")

    # Build metadata
    metadata = {
        "phase": "3.0",
        "feature_count": len(features),
        "null_count": sum(1 for v in features.values() if v is None),
        "computed_features": get_phase_30_feature_names(),
    }

    return NumericFeatureSnapshot(
        timestamp=as_of,
        features=features,
        asset_id=asset_id,
        metadata=metadata,
    )
