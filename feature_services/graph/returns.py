"""
Log return computation for graph features.

CRITICAL RULES:
- Returns = log returns: ln(P_t / P_{t-1})
- Only backward-looking data (data <= as_of)
- No forward-fill of missing prices
- Assets with insufficient history are excluded

Log returns are used because:
1. Additive across time: log(P_T/P_0) = sum of log returns
2. Symmetric: -10% and +10% treated consistently
3. Natural for correlation/covariance computation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute log returns from price data for a backward-looking window.

    Args:
        prices: DataFrame with columns = asset tickers, index = timestamps,
                values = close prices. Must be sorted by index.
        as_of: Computation reference timestamp. Only data <= as_of is used.
        window_days: Number of trading days for the window.
        min_periods: Minimum required observations per asset. Defaults to
                     window_days (strict requirement).

    Returns:
        DataFrame of log returns for valid assets only.
        Index = timestamps within window.
        Columns = asset tickers with sufficient data.

    Raises:
        ValueError: If prices is empty or has invalid structure.

    Mathematical definition:
        r_t = ln(P_t) - ln(P_{t-1})

    Economic interpretation:
        Log returns approximate percentage returns for small changes
        and have better statistical properties for aggregation.
    """
    if prices.empty:
        raise ValueError("Prices DataFrame is empty")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must have DatetimeIndex")

    if min_periods is None:
        min_periods = window_days

    # Filter to data <= as_of (prevent lookahead bias)
    prices_filtered = prices[prices.index <= as_of].copy()

    if len(prices_filtered) < 2:
        # Need at least 2 observations for returns
        return pd.DataFrame()

    # Take last (window_days + 1) observations to get window_days returns
    prices_window = prices_filtered.tail(window_days + 1)

    # Compute log returns
    log_returns = np.log(prices_window / prices_window.shift(1))

    # Drop the first NaN row from shift
    log_returns = log_returns.iloc[1:]

    # Identify assets with sufficient non-null data
    valid_assets = log_returns.columns[
        log_returns.notna().sum() >= min_periods
    ]

    if len(valid_assets) == 0:
        return pd.DataFrame()

    return log_returns[valid_assets]


def filter_valid_assets(
    returns: pd.DataFrame,
    min_overlap: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter returns to assets with overlapping valid data.

    For correlation computation, we need assets with valid data on the
    same dates. This function identifies the intersection of valid dates
    and filters to assets meeting the overlap requirement.

    Args:
        returns: DataFrame of log returns.
        min_overlap: Minimum number of overlapping observations required.
                     Defaults to len(returns) (require all dates).

    Returns:
        Tuple of:
        - Filtered returns DataFrame with only valid assets
        - List of excluded asset tickers

    Economic interpretation:
        Correlation between assets is only meaningful if computed on
        the same time periods. Excluding assets with gaps ensures
        statistical validity.
    """
    if returns.empty:
        return returns, []

    if min_overlap is None:
        min_overlap = len(returns)

    # Find assets with data on all (or enough) common dates
    valid_count = returns.notna().sum()
    valid_assets = valid_count[valid_count >= min_overlap].index.tolist()
    excluded = [c for c in returns.columns if c not in valid_assets]

    if len(valid_assets) == 0:
        return pd.DataFrame(), list(returns.columns)

    # For included assets, drop rows with any NaN
    filtered = returns[valid_assets].dropna()

    return filtered, excluded


def compute_sector_returns(
    asset_returns: pd.DataFrame,
    sector_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Compute equal-weighted sector returns from asset returns.

    Args:
        asset_returns: DataFrame of log returns, columns = asset tickers.
        sector_mapping: Dict mapping asset ticker → sector name.
                        Assets not in mapping are excluded.

    Returns:
        DataFrame of sector returns, columns = sector names.

    Mathematical definition:
        r_sector,t = (1/N) * sum(r_asset,t) for assets in sector

    Economic interpretation:
        Sector returns aggregate individual asset behavior to measure
        broader industry-level dynamics. Equal weighting avoids
        concentration bias from large-cap dominance.
    """
    if asset_returns.empty:
        return pd.DataFrame()

    # Filter to assets with known sector
    known_assets = [c for c in asset_returns.columns if c in sector_mapping]
    if len(known_assets) == 0:
        return pd.DataFrame()

    # Group by sector and compute mean
    sector_groups: dict[str, list[str]] = {}
    for asset in known_assets:
        sector = sector_mapping[asset]
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(asset)

    # Compute equal-weighted sector returns
    sector_returns_data: dict[str, pd.Series[Any]] = {}
    for sector, assets in sector_groups.items():
        sector_returns_data[sector] = asset_returns[assets].mean(axis=1)

    return pd.DataFrame(sector_returns_data)


def compute_market_returns(
    asset_returns: pd.DataFrame,
    market_weights: dict[str, float] | None = None,
) -> pd.Series[Any]:
    """
    Compute market portfolio returns.

    Args:
        asset_returns: DataFrame of log returns, columns = asset tickers.
        market_weights: Optional dict of asset → weight. If None, uses
                        equal weighting.

    Returns:
        Series of market returns.

    Mathematical definition:
        r_market,t = sum(w_i * r_i,t)
        
        If equal-weighted: w_i = 1/N for all i

    Economic interpretation:
        Market return represents the aggregate behavior of the
        asset universe. Used as benchmark for sector correlations.
    """
    if asset_returns.empty:
        return pd.Series(dtype=float)

    if market_weights is None:
        # Equal-weighted
        return asset_returns.mean(axis=1)

    # Weighted average
    weights = pd.Series(market_weights)
    common = weights.index.intersection(asset_returns.columns)
    if len(common) == 0:
        return pd.Series(dtype=float)

    w = weights[common]
    w = w / w.sum()  # Normalize to sum to 1
    return (asset_returns[common] * w).sum(axis=1)
