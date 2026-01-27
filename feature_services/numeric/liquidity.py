"""
Liquidity feature computations.

Features from features.yaml:
- volume_zscore_20d
- volume_price_divergence

All functions are pure and stateless.
Missing data returns None, never fallback values.
"""

import numpy as np
import pandas as pd

from .windows import compute_log_returns, get_window_data, rolling_zscore

# Constants from features.yaml
VOLUME_NORMALIZATION_LOOKBACK = 60
VOLUME_NORMALIZATION_MIN_PERIODS = 20


def compute_volume_zscore_20d(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
) -> float | None:
    """
    Compute z-score of current volume relative to trailing distribution.

    From features.yaml:
        source_category: market_prices
        source_fields: [volume]
        window: 20d
        normalization: rolling_zscore
        normalization_params:
            lookback: 60d
            min_periods: 20

    Mathematical definition:
        z = (volume_today - mean(volume_60d)) / std(volume_60d)

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, volume, ...]
        as_of: Reference timestamp (computation uses data <= as_of)
        asset_id: Asset to compute volume z-score for

    Returns:
        Z-score of current volume, or None if insufficient data

    Economic intuition:
        Volume spikes during price declines signal panic selling and
        potential liquidity stress. Abnormally low volume may indicate
        market dysfunction or lack of price discovery.
    """
    # Filter to specific asset
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    # Get 60-day volume history for normalization
    volumes = get_window_data(
        df, as_of, window_days=VOLUME_NORMALIZATION_LOOKBACK, column="volume"
    )

    if volumes is None or len(volumes) < VOLUME_NORMALIZATION_MIN_PERIODS:
        return None

    # Current volume is the last value
    current_volume = float(volumes.iloc[-1])

    if current_volume < 0 or np.isnan(current_volume):
        return None

    # Z-score relative to trailing 60d distribution
    # Exclude current value from historical distribution
    historical_volumes = volumes.iloc[:-1]

    return rolling_zscore(
        current_volume, historical_volumes, VOLUME_NORMALIZATION_MIN_PERIODS - 1
    )


def compute_volume_price_divergence(
    market_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    asset_id: str = "SPY",
    window_days: int = 20,
) -> float | None:
    """
    Compute divergence between price trend and volume trend.

    From features.yaml:
        source_category: market_prices
        source_fields: [close, volume]
        window: 20d
        normalization: none
        output_range: [-1.0, 1.0]

    Mathematical definition:
        divergence = correlation(returns, volume_changes) over window

    Args:
        market_prices: DataFrame with columns [timestamp, asset_id, close, volume]
        as_of: Reference timestamp
        asset_id: Asset to compute divergence for
        window_days: Window size (default 20)

    Returns:
        Correlation coefficient [-1, 1], or None if insufficient data

    Economic intuition:
        Negative divergence (high volume on down days) indicates distribution
        and potential for cascading sells. Positive divergence (high volume
        on up days) indicates accumulation and healthy buying pressure.
    """
    # Filter to specific asset
    if "asset_id" in market_prices.columns:
        df = market_prices[market_prices["asset_id"] == asset_id]
    else:
        df = market_prices

    # Need window_days + 1 to compute window_days returns
    prices = get_window_data(df, as_of, window_days=window_days + 1, column="close")
    volumes = get_window_data(df, as_of, window_days=window_days + 1, column="volume")

    if prices is None or volumes is None:
        return None

    if len(prices) < window_days + 1 or len(volumes) < window_days + 1:
        return None

    # Compute returns and volume changes
    returns = compute_log_returns(prices)
    volume_changes = volumes.diff().dropna()

    if returns is None or len(returns) < window_days:
        return None

    if len(volume_changes) < window_days:
        return None

    # Align arrays (take last window_days)
    returns_arr = returns.values[-window_days:]
    volume_changes_arr = volume_changes.values[-window_days:]

    # Check for valid data
    if np.any(np.isnan(returns_arr)) or np.any(np.isnan(volume_changes_arr)):
        return None

    # Compute correlation
    if np.std(returns_arr) == 0 or np.std(volume_changes_arr) == 0:
        return None

    correlation = np.corrcoef(returns_arr, volume_changes_arr)[0, 1]

    if np.isnan(correlation):
        return None

    # Clamp to [-1, 1] to handle floating point issues
    correlation = float(np.clip(correlation, -1.0, 1.0))

    return correlation
