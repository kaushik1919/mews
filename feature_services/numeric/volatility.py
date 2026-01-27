"""
Volatility feature computations.

Features from features.yaml:
- realized_volatility_20d
- realized_volatility_60d
- volatility_ratio_20d_60d
- vix_level

All functions are pure and stateless.
Missing data returns None, never fallback values.
"""

import numpy as np
import pandas as pd

from .windows import (
    compute_log_returns,
    get_window_data,
    rolling_zscore,
)

# Constants from features.yaml
ANNUALIZATION_FACTOR = np.sqrt(252)  # Trading days per year
NORMALIZATION_LOOKBACK_DAYS = 252
NORMALIZATION_MIN_PERIODS = 63


def realized_volatility(
    prices: pd.Series,
    window_days: int,
) -> float | None:
    """
    Compute annualized realized volatility from price series.

    Mathematical definition:
        σ = std(log_returns) * sqrt(252)

    Args:
        prices: Series of closing prices (length = window_days)
        window_days: Window size (for validation)

    Returns:
        Annualized volatility as decimal, or None if insufficient data

    Economic intuition:
        Measures the dispersion of returns over the window period.
        Higher values indicate greater price uncertainty and risk.
        Annualization allows comparison across different time horizons.
    """
    if prices is None or len(prices) < window_days:
        return None

    log_rets = compute_log_returns(prices)
    if log_rets is None or len(log_rets) < window_days - 1:
        return None

    # Standard deviation of log returns, annualized
    vol = float(log_rets.std() * ANNUALIZATION_FACTOR)

    # Volatility must be non-negative
    if vol < 0 or np.isnan(vol):
        return None

    return vol


def compute_realized_volatility_20d(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
    normalize: bool = False,
    normalization_history: pd.Series | None = None,
) -> float | None:
    """
    Compute 20-day realized volatility for an asset.

    From features.yaml:
        source_category: market_prices
        source_fields: [close]
        window: 20d
        normalization: rolling_zscore
        normalization_params:
            lookback: 252d
            min_periods: 63

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, close, ...]
        as_of: Reference timestamp (computation uses data <= as_of)
        asset_id: Asset to compute volatility for
        normalize: Whether to apply z-score normalization
        normalization_history: Pre-computed historical volatilities for z-score

    Returns:
        Annualized 20-day volatility (raw or z-scored), or None

    Economic intuition:
        Short-term volatility captures immediate market stress.
        Spikes often precede or accompany crisis events.
    """
    # Filter to specific asset
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    prices = get_window_data(df, as_of, window_days=20, column="close")
    vol = realized_volatility(prices, window_days=20)

    if vol is None:
        return None

    if normalize and normalization_history is not None:
        return rolling_zscore(vol, normalization_history, NORMALIZATION_MIN_PERIODS)

    return vol


def compute_realized_volatility_60d(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
    normalize: bool = False,
    normalization_history: pd.Series | None = None,
) -> float | None:
    """
    Compute 60-day realized volatility for an asset.

    From features.yaml:
        source_category: market_prices
        source_fields: [close]
        window: 60d
        normalization: rolling_zscore

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, close, ...]
        as_of: Reference timestamp (computation uses data <= as_of)
        asset_id: Asset to compute volatility for
        normalize: Whether to apply z-score normalization
        normalization_history: Pre-computed historical volatilities for z-score

    Returns:
        Annualized 60-day volatility (raw or z-scored), or None

    Economic intuition:
        Medium-term volatility is less sensitive to single-day shocks.
        Persistent elevation indicates regime shift rather than transient stress.
    """
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    prices = get_window_data(df, as_of, window_days=60, column="close")
    vol = realized_volatility(prices, window_days=60)

    if vol is None:
        return None

    if normalize and normalization_history is not None:
        return rolling_zscore(vol, normalization_history, NORMALIZATION_MIN_PERIODS)

    return vol


def compute_volatility_ratio(
    vol_20d: float | None,
    vol_60d: float | None,
) -> float | None:
    """
    Compute ratio of 20-day to 60-day volatility.

    From features.yaml:
        source_category: derived
        derived_from: [realized_volatility_20d, realized_volatility_60d]
        normalization: none

    Args:
        vol_20d: 20-day realized volatility (raw, not z-scored)
        vol_60d: 60-day realized volatility (raw, not z-scored)

    Returns:
        Ratio of short to medium-term volatility, or None

    Mathematical definition:
        ratio = vol_20d / vol_60d

    Economic intuition:
        Values > 1 indicate short-term stress exceeding baseline.
        Spikes signal acute episodes; sustained elevation indicates propagation.
    """
    if vol_20d is None or vol_60d is None:
        return None

    if vol_60d == 0:
        return None

    ratio = vol_20d / vol_60d

    if np.isnan(ratio) or np.isinf(ratio):
        return None

    return float(ratio)


def compute_vix_level(
    volatility_indices: pd.DataFrame,
    as_of: pd.Timestamp,
    index_id: str = "^VIX",
    normalize: bool = False,
    normalization_history: pd.Series | None = None,
) -> float | None:
    """
    Get VIX level as of timestamp.

    From features.yaml:
        source_category: volatility_indices
        source_fields: [close]
        window: 1d
        normalization: rolling_zscore
        normalization_params:
            lookback: 252d
            min_periods: 63

    Args:
        volatility_indices: DataFrame with columns [timestamp, index_id, close]
        as_of: Reference timestamp
        index_id: VIX index identifier (default: ^VIX)
        normalize: Whether to apply z-score normalization
        normalization_history: Pre-computed historical VIX values for z-score

    Returns:
        VIX level (raw or z-scored), or None

    Economic intuition:
        VIX > 30 historically associated with crisis conditions.
        Captures market-implied fear and hedging demand.
        Forward-looking indicator derived from options prices.
    """
    # Filter to VIX index
    id_col = "index_id" if "index_id" in volatility_indices.columns else "asset_id"
    df = volatility_indices[volatility_indices[id_col] == index_id]

    # Get single day value (window = 1d)
    prices = get_window_data(df, as_of, window_days=1, column="close")

    if prices is None or len(prices) < 1:
        return None

    vix_value = float(prices.iloc[-1])

    if vix_value < 0 or np.isnan(vix_value):
        return None

    if normalize and normalization_history is not None:
        return rolling_zscore(
            vix_value, normalization_history, NORMALIZATION_MIN_PERIODS
        )

    return vix_value
