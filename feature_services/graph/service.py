"""
Graph Feature Service - Public API.

Converts cross-asset relationships into scalar measures of systemic coupling
and fragility. This service measures how tightly the financial system is
wired together.

NOT for:
- Predicting returns
- Learning graphs
- Optimizing portfolios

FOR:
- Measuring systemic connectivity
- Detecting correlation regime changes
- Quantifying diversification failure

Public API:
    compute_graph_features(market_prices, as_of) -> GraphFeatureSnapshot
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from feature_services.graph.correlation import (
    compute_avg_pairwise_correlation,
    compute_correlation_dispersion,
    compute_correlation_matrix,
    compute_mean_sector_correlation,
    compute_sector_to_market_correlations,
)
from feature_services.graph.network import (
    compute_centrality_change,
    compute_degree_centrality,
)
from feature_services.graph.returns import (
    compute_log_returns,
    compute_market_returns,
    compute_sector_returns,
    filter_valid_assets,
)
from feature_services.graph.validate import (
    get_phase_32_feature_names,
    validate_feature_snapshot,
    validate_input_market_prices,
    validate_no_future_data,
)

# Default window sizes
CORRELATION_WINDOW_DAYS = 20
CENTRALITY_SHIFT_DAYS = 20  # Gap between current and previous window


@dataclass
class GraphFeatureSnapshot:
    """
    Point-in-time snapshot of graph features.

    Attributes:
        timestamp: as_of timestamp for this snapshot
        features: Dict of feature_name -> value (or None if unavailable)
        asset_count: Number of valid assets used
        is_complete: True if all features computed successfully
        computation_metadata: Additional info about the computation
    """

    timestamp: pd.Timestamp
    features: dict[str, float | None]
    asset_count: int
    is_complete: bool
    computation_metadata: dict[str, Any] = field(default_factory=dict)


def compute_graph_features(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    sector_mapping: dict[str, str] | None = None,
    window_days: int = CORRELATION_WINDOW_DAYS,
) -> GraphFeatureSnapshot:
    """
    Compute graph-based systemic risk features.

    Args:
        market_prices: DataFrame with columns = asset tickers,
                       index = timestamps (DatetimeIndex),
                       values = close prices.
        as_of: Computation reference timestamp. Only data <= as_of is used.
        sector_mapping: Optional dict of asset ticker -> sector name.
                        If None, sector features will be null.
        window_days: Lookback window for correlation computation.

    Returns:
        GraphFeatureSnapshot with all Phase 3.2 features.

    Features computed:
        - avg_pairwise_correlation_20d: Mean off-diagonal correlation
        - correlation_dispersion_20d: Std dev of correlations
        - sector_correlation_to_market: Mean sector-to-market correlation
        - network_centrality_change: Mean absolute change in degree centrality

    Raises:
        ValueError: If input validation fails.

    Temporal integrity:
        - Only data with timestamp <= as_of is used
        - All windows are backward-looking
        - No forward-fill, no imputation
    """
    # Initialize null features
    null_features = dict.fromkeys(get_phase_32_feature_names())

    # Validate inputs
    valid, errors = validate_input_market_prices(market_prices)
    if not valid:
        return GraphFeatureSnapshot(
            timestamp=as_of,
            features=null_features,
            asset_count=0,
            is_complete=False,
            computation_metadata={"validation_errors": errors},
        )

    # Filter to data <= as_of
    prices_filtered = market_prices[market_prices.index <= as_of].copy()

    # Validate no future data
    valid, errors = validate_no_future_data(prices_filtered, as_of)
    if not valid:
        return GraphFeatureSnapshot(
            timestamp=as_of,
            features=null_features,
            asset_count=0,
            is_complete=False,
            computation_metadata={"validation_errors": errors},
        )

    # Compute current window returns
    returns_current = compute_log_returns(
        prices_filtered,
        as_of=as_of,
        window_days=window_days,
    )

    if returns_current.empty:
        return GraphFeatureSnapshot(
            timestamp=as_of,
            features=null_features,
            asset_count=0,
            is_complete=False,
            computation_metadata={"error": "Insufficient data for returns"},
        )

    # Filter to valid assets with overlapping data
    returns_valid, excluded = filter_valid_assets(returns_current)

    if returns_valid.empty:
        return GraphFeatureSnapshot(
            timestamp=as_of,
            features=null_features,
            asset_count=0,
            is_complete=False,
            computation_metadata={"error": "No valid asset overlap"},
        )

    asset_count = len(returns_valid.columns)

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(returns_valid)

    if corr_matrix is None:
        return GraphFeatureSnapshot(
            timestamp=as_of,
            features=null_features,
            asset_count=asset_count,
            is_complete=False,
            computation_metadata={"error": "Invalid correlation matrix"},
        )

    # Compute features
    features = _compute_all_features(
        returns_valid=returns_valid,
        corr_matrix=corr_matrix,
        prices_filtered=prices_filtered,
        as_of=as_of,
        window_days=window_days,
        sector_mapping=sector_mapping,
    )

    # Validate output
    valid, errors = validate_feature_snapshot(features, strict=True)
    is_complete = all(v is not None for v in features.values())

    return GraphFeatureSnapshot(
        timestamp=as_of,
        features=features,
        asset_count=asset_count,
        is_complete=is_complete,
        computation_metadata={
            "excluded_assets": excluded,
            "validation_errors": errors if not valid else [],
        },
    )


def _compute_all_features(
    returns_valid: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    prices_filtered: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
    sector_mapping: dict[str, str] | None,
) -> dict[str, float | None]:
    """
    Compute all Phase 3.2 graph features.

    Internal function called by compute_graph_features.
    """
    features: dict[str, float | None] = {}

    # 1. Average pairwise correlation
    features["avg_pairwise_correlation_20d"] = compute_avg_pairwise_correlation(
        corr_matrix
    )

    # 2. Correlation dispersion
    features["correlation_dispersion_20d"] = compute_correlation_dispersion(
        corr_matrix
    )

    # 3. Sector correlation to market
    features["sector_correlation_to_market"] = _compute_sector_feature(
        returns_valid=returns_valid,
        sector_mapping=sector_mapping,
    )

    # 4. Network centrality change
    features["network_centrality_change"] = _compute_centrality_change_feature(
        prices_filtered=prices_filtered,
        as_of=as_of,
        window_days=window_days,
        current_corr=corr_matrix,
    )

    return features


def _compute_sector_feature(
    returns_valid: pd.DataFrame,
    sector_mapping: dict[str, str] | None,
) -> float | None:
    """
    Compute sector_correlation_to_market feature.

    Steps:
    1. Compute equal-weighted sector returns
    2. Compute market return (equal-weighted across all assets)
    3. Compute correlation of each sector to market
    4. Return mean sector-to-market correlation
    """
    if sector_mapping is None or not sector_mapping:
        return None

    # Compute sector returns
    sector_returns = compute_sector_returns(returns_valid, sector_mapping)
    if sector_returns.empty:
        return None

    # Compute market return
    market_return = compute_market_returns(returns_valid)
    if market_return.empty:
        return None

    # Compute sector-to-market correlations
    sector_corrs = compute_sector_to_market_correlations(
        sector_returns, market_return
    )

    # Return mean
    return compute_mean_sector_correlation(sector_corrs)


def _compute_centrality_change_feature(
    prices_filtered: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
    current_corr: pd.DataFrame,
) -> float | None:
    """
    Compute network_centrality_change feature.

    Steps:
    1. Compute current degree centrality from current correlation matrix
    2. Compute previous window returns (shifted by window_days)
    3. Compute previous correlation matrix
    4. Compute previous degree centrality
    5. Return mean absolute change

    This measures how network structure is shifting over time.
    Rapid changes indicate regime shifts in market connectivity.
    """
    # Current centrality
    current_centrality = compute_degree_centrality(current_corr, use_absolute=True)
    if current_centrality.empty:
        return None

    # Compute previous window
    # Previous window ends (window_days) before the current window starts
    # Current window: [as_of - window_days + 1, as_of]
    # Previous window: [as_of - 2*window_days + 1, as_of - window_days]
    previous_as_of = as_of - pd.Timedelta(days=window_days)

    previous_returns = compute_log_returns(
        prices_filtered,
        as_of=previous_as_of,
        window_days=window_days,
    )

    if previous_returns.empty:
        return None

    previous_valid, _ = filter_valid_assets(previous_returns)
    if previous_valid.empty:
        return None

    previous_corr = compute_correlation_matrix(previous_valid)
    if previous_corr is None:
        return None

    previous_centrality = compute_degree_centrality(previous_corr, use_absolute=True)
    if previous_centrality.empty:
        return None

    # Compute change
    return compute_centrality_change(current_centrality, previous_centrality)
