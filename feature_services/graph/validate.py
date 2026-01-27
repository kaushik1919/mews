"""
Graph feature validation against core_specs/features.yaml.

Ensures:
1. Feature names match spec exactly
2. No extra features
3. No missing required features
4. No NaN values (use None)
5. All input data has timestamp <= as_of

Fail fast on violations.
"""

from importlib import resources
from typing import Any

import pandas as pd
import yaml

# Minimum assets required for meaningful graph analysis
MIN_ASSETS_REQUIRED = 5

# Minimum observations required per asset
MIN_OBSERVATIONS_REQUIRED = 10


def load_feature_spec() -> dict[str, Any]:
    """Load features.yaml specification."""
    try:
        with resources.files("core_specs").joinpath("features.yaml").open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Feature spec not found. Ensure core_specs package is installed."
        ) from None


def get_graph_feature_names() -> list[str]:
    """
    Get list of all graph feature names from spec.

    Returns all graph features defined in features.yaml.
    """
    spec = load_feature_spec()
    graph = spec.get("graph", {})
    return list(graph.keys())


def get_phase_32_feature_names() -> list[str]:
    """
    Get list of graph features implemented in Phase 3.2.

    Phase 3.2 scope:
    - avg_pairwise_correlation_20d
    - correlation_dispersion_20d
    - sector_correlation_to_market
    - network_centrality_change
    """
    return [
        "avg_pairwise_correlation_20d",
        "correlation_dispersion_20d",
        "sector_correlation_to_market",
        "network_centrality_change",
    ]


def validate_feature_snapshot(
    features: dict[str, float | None],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate a graph feature snapshot against spec.

    Args:
        features: Dict of feature_name -> value (or None)
        strict: If True, require exact match to Phase 3.2 features

    Returns:
        Tuple of (is_valid, list of error messages)

    Validation rules:
    1. All feature names must exist in features.yaml
    2. If strict, must have exactly Phase 3.2 features
    3. No NaN values (use None instead)
    4. Values must be in valid range if not None
    """
    errors = []
    spec_features = set(get_graph_feature_names())
    phase_32_features = set(get_phase_32_feature_names())

    # Check feature names
    for name in features:
        if name not in spec_features:
            errors.append(f"Unknown feature: {name} (not in features.yaml)")

    # Check for required features if strict
    if strict:
        missing = phase_32_features - set(features.keys())
        if missing:
            errors.append(f"Missing Phase 3.2 features: {sorted(missing)}")

        extra = set(features.keys()) - phase_32_features
        if extra:
            errors.append(f"Extra features not in Phase 3.2: {sorted(extra)}")

    # Check for NaN values
    import math
    for name, value in features.items():
        if value is not None and math.isnan(value):
            errors.append(f"Feature {name} has NaN value (use None instead)")

    # Validate value ranges
    for name, value in features.items():
        if value is None:
            continue

        # Correlation features should be in [-1, 1]
        if "correlation" in name and not name.endswith("change"):
            if not -1.0 <= value <= 1.0:
                errors.append(
                    f"Feature {name} out of range: {value} (expected [-1, 1])"
                )

        # Dispersion should be non-negative
        if "dispersion" in name:
            if value < 0.0:
                errors.append(
                    f"Feature {name} is negative: {value} (expected >= 0)"
                )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_input_market_prices(
    market_prices: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Validate market prices input for graph features.

    Args:
        market_prices: DataFrame of prices (columns=assets, index=timestamps)

    Returns:
        Tuple of (is_valid, list of error messages)

    Validation rules:
    1. Must be a DataFrame
    2. Must have DatetimeIndex
    3. Must have at least MIN_ASSETS_REQUIRED columns (assets)
    4. Must have at least MIN_OBSERVATIONS_REQUIRED rows
    5. Prices must be positive
    """
    errors = []

    if not isinstance(market_prices, pd.DataFrame):
        errors.append("market_prices must be a pandas DataFrame")
        return False, errors

    if market_prices.empty:
        errors.append("market_prices DataFrame is empty")
        return False, errors

    if not isinstance(market_prices.index, pd.DatetimeIndex):
        errors.append("market_prices must have DatetimeIndex")

    n_assets = len(market_prices.columns)
    if n_assets < MIN_ASSETS_REQUIRED:
        errors.append(
            f"Insufficient assets: {n_assets} (minimum {MIN_ASSETS_REQUIRED})"
        )

    n_obs = len(market_prices)
    if n_obs < MIN_OBSERVATIONS_REQUIRED:
        errors.append(
            f"Insufficient observations: {n_obs} (minimum {MIN_OBSERVATIONS_REQUIRED})"
        )

    # Check for non-positive prices
    if (market_prices <= 0).any().any():
        errors.append("Prices must be positive (found zero or negative values)")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_no_future_data(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[bool, list[str]]:
    """
    Validate that no future data is used.

    Args:
        market_prices: DataFrame of prices
        as_of: Reference timestamp

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if market_prices.empty:
        return True, []

    max_timestamp = market_prices.index.max()
    if max_timestamp > as_of:
        errors.append(
            f"Future data detected: max timestamp {max_timestamp} > as_of {as_of}"
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_sector_mapping(
    sector_mapping: dict[str, str] | None,
    asset_tickers: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate sector mapping configuration.

    Args:
        sector_mapping: Dict of asset -> sector
        asset_tickers: List of asset tickers in the data

    Returns:
        Tuple of (is_valid, list of warnings)

    Note: Missing mappings are warnings, not errors.
    """
    warnings = []

    if sector_mapping is None:
        warnings.append("No sector mapping provided; sector features will be null")
        return True, warnings

    if not sector_mapping:
        warnings.append("Sector mapping is empty; sector features will be null")
        return True, warnings

    # Check coverage
    mapped = set(sector_mapping.keys())
    assets = set(asset_tickers)
    unmapped = assets - mapped
    if unmapped:
        warnings.append(
            f"{len(unmapped)} assets have no sector mapping: "
            f"{sorted(list(unmapped)[:5])}{'...' if len(unmapped) > 5 else ''}"
        )

    return True, warnings


def get_default_sector_mapping() -> dict[str, str]:
    """
    Get default sector mapping for common assets.

    This is a minimal mapping for demonstration/testing.
    Production systems should load from configuration.

    Returns:
        Dict of asset ticker -> GICS sector name
    """
    # Minimal mapping for major indices/ETFs
    return {
        # Sector ETFs
        "XLF": "Financials",
        "XLK": "Technology",
        "XLE": "Energy",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
        # Major stocks (examples)
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "JNJ": "Health Care",
        "PFE": "Health Care",
    }
