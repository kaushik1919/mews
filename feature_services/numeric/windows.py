"""
Rolling window computation helpers.

All helpers operate on backward-looking windows only.
No forward-fill, no future data access.

CRITICAL: These functions enforce temporal integrity.
If required data is missing, return None - never fallback values.
"""


import numpy as np
import pandas as pd


def get_window_data(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
    column: str,
    date_column: str = "timestamp",
) -> pd.Series | None:
    """
    Extract backward-looking window data up to and including as_of date.

    Args:
        df: DataFrame with time-indexed data
        as_of: Reference timestamp (inclusive upper bound)
        window_days: Number of trading days to look back
        column: Column to extract
        date_column: Name of date/timestamp column

    Returns:
        Series of values within window, or None if insufficient data

    CRITICAL: Only data with timestamp <= as_of is included.
    This prevents lookahead bias.
    """
    if df is None or df.empty:
        return None

    if column not in df.columns:
        return None

    if date_column not in df.columns:
        return None

    # Ensure as_of is timezone-aware for comparison
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    # Filter to data <= as_of (no future data)
    df_ts = df[date_column]
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")

    mask = df_ts <= as_of_ts
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return None

    # Sort by timestamp and take last window_days
    df_filtered = df_filtered.sort_values(date_column)
    df_window = df_filtered.tail(window_days)

    if len(df_window) < window_days:
        # Insufficient data for complete window
        return None

    return df_window[column].reset_index(drop=True)


def compute_log_returns(prices: pd.Series) -> pd.Series | None:
    """
    Compute log returns from price series.

    Args:
        prices: Series of prices

    Returns:
        Series of log returns (length = len(prices) - 1), or None if invalid

    Mathematical definition:
        r_t = ln(P_t / P_{t-1})

    Economic intuition:
        Log returns are additive over time and approximately equal to
        simple returns for small changes. Preferred for volatility calculations.
    """
    if prices is None or len(prices) < 2:
        return None

    prices_arr = prices.values.astype(float)

    # Check for non-positive prices (invalid for log)
    if np.any(prices_arr <= 0):
        return None

    log_returns = np.diff(np.log(prices_arr))
    return pd.Series(log_returns)


def rolling_zscore(
    value: float,
    historical_series: pd.Series,
    min_periods: int,
) -> float | None:
    """
    Compute z-score of value relative to historical distribution.

    Args:
        value: Current value to normalize
        historical_series: Historical values for distribution
        min_periods: Minimum required observations

    Returns:
        Z-score or None if insufficient data

    Mathematical definition:
        z = (x - μ) / σ

    Economic intuition:
        Measures how unusual the current value is relative to recent history.
        Values > 2 indicate unusual stress; values < -2 indicate unusual calm.

    CRITICAL: Uses only historical data - no future information.
    """
    if historical_series is None or len(historical_series) < min_periods:
        return None

    if value is None or np.isnan(value):
        return None

    mean = historical_series.mean()
    std = historical_series.std()

    if std == 0 or np.isnan(std):
        return None

    return float((value - mean) / std)


def get_normalization_window(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
    column: str,
    date_column: str = "timestamp",
) -> pd.Series | None:
    """
    Get data for normalization (longer lookback than feature window).

    Args:
        df: DataFrame with time-indexed data
        as_of: Reference timestamp
        lookback_days: Normalization lookback period
        column: Column for normalization values
        date_column: Name of date column

    Returns:
        Series of historical values, or None if unavailable

    NOTE: Unlike get_window_data, this returns available data even if
    less than lookback_days, as long as min_periods is satisfied.
    Caller must check min_periods requirement.
    """
    if df is None or df.empty:
        return None

    if column not in df.columns:
        return None

    # Ensure timezone awareness
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    df_ts = df[date_column]
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")

    mask = df_ts <= as_of_ts
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return None

    # Sort and take last lookback_days
    df_filtered = df_filtered.sort_values(date_column)
    df_window = df_filtered.tail(lookback_days)

    return df_window[column].reset_index(drop=True)
