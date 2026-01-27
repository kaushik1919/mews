"""
Sentiment aggregation module.

Aggregates article-level sentiment scores to daily and rolling features.

CRITICAL TEMPORAL RULES:
1. Use aligned_timestamp only (not publication timestamp)
2. Include only articles where aligned_timestamp <= as_of
3. No forward-looking articles
4. No forward-fill of sentiment
5. Missing data → explicit null

From features.yaml:
- news_sentiment_daily: mean of article scores for day
- news_sentiment_5d: exponential weighted mean over 5 days
- sentiment_volatility_20d: std dev over 20 days
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd


def aggregate_daily_sentiment(
    article_scores: pd.DataFrame,
    target_date: date,
) -> float | None:
    """
    Compute daily aggregate sentiment from article-level scores.

    Args:
        article_scores: DataFrame with columns [aligned_date, score]
        target_date: Date to compute sentiment for

    Returns:
        Mean sentiment score for the day, or None if no articles

    Mathematical definition:
        daily_sentiment(T) = mean(score_i for all articles aligned to T)

    Economic intuition:
        Equal weighting treats all articles as equal signal sources.
        Mean aggregation cancels noise while preserving directional signal.

    CRITICAL: Only uses articles with aligned_date == target_date.
    """
    if article_scores is None or article_scores.empty:
        return None

    # Filter to target date
    day_articles = article_scores[article_scores["aligned_date"] == target_date]

    if day_articles.empty:
        return None

    # Compute mean sentiment
    scores = day_articles["score"].dropna()

    if len(scores) == 0:
        return None

    return float(scores.mean())


def compute_daily_sentiment_series(
    article_scores: pd.DataFrame,
    end_date: date,
    lookback_days: int,
) -> pd.Series:
    """
    Compute daily sentiment for a range of dates.

    Args:
        article_scores: DataFrame with columns [aligned_date, score]
        end_date: Last date to compute (inclusive)
        lookback_days: Number of days to look back

    Returns:
        Series indexed by date with daily sentiment values
        Missing days have NaN (not None, for pandas compatibility)
    """
    if article_scores is None or article_scores.empty:
        return pd.Series(dtype=float)

    # Generate date range
    start_date = end_date - timedelta(days=lookback_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Compute daily sentiment for each date
    daily_values = {}
    for d in dates:
        d_date = d.date()
        sentiment = aggregate_daily_sentiment(article_scores, d_date)
        daily_values[d_date] = sentiment if sentiment is not None else np.nan

    return pd.Series(daily_values)


def compute_rolling_sentiment(
    daily_sentiment: pd.Series,
    window_days: int,
    use_ewm: bool = False,
    ewm_span: int | None = None,
) -> float | None:
    """
    Compute rolling sentiment from daily values.

    Args:
        daily_sentiment: Series of daily sentiment values
        window_days: Window size
        use_ewm: If True, use exponential weighted mean
        ewm_span: Span for EWM (defaults to window_days)

    Returns:
        Rolling sentiment value, or None if insufficient data

    Mathematical definition:
        If use_ewm=False: mean of last window_days values
        If use_ewm=True: exponentially weighted mean with span

    Economic intuition:
        EWM gives more weight to recent sentiment, capturing
        momentum in narrative shifts while smoothing noise.

    CRITICAL: Requires full window of non-null values.
    """
    if daily_sentiment is None or len(daily_sentiment) < window_days:
        return None

    # Take last window_days
    window = daily_sentiment.tail(window_days)

    # Check for sufficient non-null values
    valid_count = window.notna().sum()
    if valid_count < window_days:
        # Require full window
        return None

    if use_ewm:
        span = ewm_span if ewm_span is not None else window_days
        result = window.ewm(span=span, adjust=True).mean().iloc[-1]
    else:
        result = window.mean()

    if pd.isna(result):
        return None

    return float(result)


def compute_sentiment_volatility(
    daily_sentiment: pd.Series,
    window_days: int,
) -> float | None:
    """
    Compute sentiment volatility (std dev of daily sentiment).

    Args:
        daily_sentiment: Series of daily sentiment values
        window_days: Window size

    Returns:
        Standard deviation of sentiment, or None if insufficient data

    Mathematical definition:
        sentiment_vol(T) = std(daily_sentiment[T-window+1:T])

    Economic intuition:
        High volatility indicates market confusion and conflicting
        narratives. Low volatility indicates consensus (bullish or bearish).

    CRITICAL: Requires full window of non-null values.
    """
    if daily_sentiment is None or len(daily_sentiment) < window_days:
        return None

    # Take last window_days
    window = daily_sentiment.tail(window_days)

    # Check for sufficient non-null values
    valid_count = window.notna().sum()
    if valid_count < window_days:
        return None

    result = window.std()

    if pd.isna(result):
        return None

    # Volatility must be non-negative
    return max(0.0, float(result))


def filter_articles_by_as_of(
    news_events: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """
    Filter news events to only include those aligned before or at as_of.

    Args:
        news_events: DataFrame with aligned_timestamp column
        as_of: Reference timestamp

    Returns:
        Filtered DataFrame

    CRITICAL: This is the primary leakage prevention mechanism.
    Only articles with aligned_timestamp <= as_of are included.
    """
    if news_events is None or news_events.empty:
        return pd.DataFrame()

    # Determine timestamp column name
    if "aligned_timestamp" in news_events.columns:
        ts_col = "aligned_timestamp"
    elif "timestamp" in news_events.columns:
        ts_col = "timestamp"
    else:
        raise ValueError("DataFrame must have 'aligned_timestamp' or 'timestamp' column")

    # Ensure timezone awareness
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    df_ts = news_events[ts_col]
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")

    # Filter to articles aligned at or before as_of
    mask = df_ts <= as_of_ts

    return news_events[mask].copy()
