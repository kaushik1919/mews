"""
Drawdown feature computations.

Features from features.yaml:
- max_drawdown_20d
- max_drawdown_60d

All functions are pure and stateless.
Missing data returns None, never fallback values.
"""

import numpy as np
import pandas as pd

from .windows import get_window_data


def compute_max_drawdown(
    prices: pd.Series,
    window_days: int,
) -> float | None:
    """
    Compute maximum peak-to-trough decline over window.

    Mathematical definition:
        drawdown_t = (price_t - max(price_0..t)) / max(price_0..t)
        max_drawdown = min(drawdown_t for t in window)

    Args:
        prices: Series of closing prices
        window_days: Window size for validation

    Returns:
        Maximum drawdown as negative decimal (e.g., -0.15 = 15% decline),
        or None if insufficient data

    Economic intuition:
        Severe drawdowns trigger margin calls, forced liquidations,
        and behavioral contagion. Threshold effects at -10%, -20% levels.
        This metric captures the worst-case loss from any peak within the window.
    """
    if prices is None or len(prices) < window_days:
        return None

    prices_arr = prices.values.astype(float)

    # Check for valid prices
    if np.any(prices_arr <= 0) or np.any(np.isnan(prices_arr)):
        return None

    # Compute running maximum (high water mark)
    running_max = np.maximum.accumulate(prices_arr)

    # Compute drawdown at each point
    drawdowns = (prices_arr - running_max) / running_max

    # Maximum drawdown is the minimum (most negative) value
    max_dd = float(np.min(drawdowns))

    # Validate result
    if np.isnan(max_dd):
        return None

    # Drawdown should be <= 0
    if max_dd > 0:
        max_dd = 0.0

    return max_dd


def compute_max_drawdown_20d(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
) -> float | None:
    """
    Compute 20-day maximum drawdown for an asset.

    From features.yaml:
        source_category: market_prices
        source_fields: [close]
        window: 20d
        normalization: none
        output_range: [-1.0, 0.0]

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, close, ...]
        as_of: Reference timestamp (computation uses data <= as_of)
        asset_id: Asset to compute drawdown for

    Returns:
        Maximum drawdown as negative decimal, or None

    Economic intuition:
        Short-term drawdowns capture immediate corrections and panic.
        Used to identify rapid de-risking events.
    """
    # Filter to specific asset
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    prices = get_window_data(df, as_of, window_days=20, column="close")
    return compute_max_drawdown(prices, window_days=20)


def compute_max_drawdown_60d(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
) -> float | None:
    """
    Compute 60-day maximum drawdown for an asset.

    From features.yaml:
        source_category: market_prices
        source_fields: [close]
        window: 60d
        normalization: none
        output_range: [-1.0, 0.0]

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, close, ...]
        as_of: Reference timestamp (computation uses data <= as_of)
        asset_id: Asset to compute drawdown for

    Returns:
        Maximum drawdown as negative decimal, or None

    Economic intuition:
        Medium-term drawdowns capture bear market conditions
        and sustained stress beyond short-term corrections.
    """
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    prices = get_window_data(df, as_of, window_days=60, column="close")
    return compute_max_drawdown(prices, window_days=60)
