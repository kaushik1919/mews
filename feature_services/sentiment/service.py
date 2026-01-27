"""
Sentiment Feature Service - Public API.

This is the main entry point for computing sentiment-based systemic risk features.
The service is STATELESS - no caching, no persistence, pure functions.

Usage:
    from feature_services.sentiment import compute_sentiment_features

    snapshot = compute_sentiment_features(
        news_events=df_news,
        as_of=pd.Timestamp("2024-01-15 21:00:00", tz="UTC"),
    )

Output format:
    {
        "timestamp": "2024-01-15T21:00:00+00:00",
        "features": {
            "news_sentiment_daily": 0.15,
            "news_sentiment_5d": 0.08,
            "sentiment_volatility_20d": 0.42,
        }
    }

Model: FinBERT (frozen, deterministic)
- No fine-tuning
- No online learning
- Mock mode available for testing
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .aggregation import (
    compute_daily_sentiment_series,
    compute_rolling_sentiment,
    compute_sentiment_volatility,
    filter_articles_by_as_of,
)
from .inference import FinBERTInference, SentimentResult
from .validate import (
    get_phase_31_feature_names,
    validate_feature_snapshot,
    validate_input_news_events,
    validate_no_future_data,
)


@dataclass
class SentimentFeatureSnapshot:
    """
    A point-in-time snapshot of sentiment features.

    Attributes:
        timestamp: The as_of timestamp (UTC)
        features: Dict of feature_name -> value (or None if missing data)
        article_count: Number of articles used in computation
        metadata: Optional computation metadata
    """

    timestamp: pd.Timestamp
    features: dict[str, float | None]
    article_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "article_count": self.article_count,
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


def compute_sentiment_features(
    news_events: pd.DataFrame,
    as_of: pd.Timestamp,
    use_mock_inference: bool = True,
    validate: bool = True,
) -> SentimentFeatureSnapshot:
    """
    Compute all Phase 3.1 sentiment features as of a specific timestamp.

    This function is STATELESS and PURE:
    - No caching or persistence
    - No side effects
    - Deterministic: same inputs -> same outputs
    - Only uses data with aligned_timestamp <= as_of (no lookahead)

    Args:
        news_events: DataFrame with Phase 2.3 output schema
            Required columns: article_id, timestamp/aligned_timestamp, headline
            Optional: body, source
        as_of: Reference timestamp (must be UTC-aware)
            All computations use only data with aligned_timestamp <= as_of
        use_mock_inference: If True, use deterministic mock FinBERT
            If False, use real FinBERT (requires transformers+torch)
        validate: Whether to validate inputs and outputs (default: True)

    Returns:
        SentimentFeatureSnapshot with all Phase 3.1 features
        Missing data -> value = None

    Raises:
        ValueError: If validation fails and validate=True

    Feature list (Phase 3.1):
        - news_sentiment_daily: Mean sentiment of articles on target day
        - news_sentiment_5d: 5-day EWM of daily sentiment
        - sentiment_volatility_20d: 20-day std dev of daily sentiment
    """
    # Ensure as_of is timezone-aware
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (UTC)")

    # Validate inputs
    if validate:
        is_valid, errors = validate_input_news_events(news_events)
        if not is_valid:
            raise ValueError(f"Invalid input news_events: {errors}")

    # Handle empty DataFrame
    if news_events is None or news_events.empty:
        features = {name: None for name in get_phase_31_feature_names()}
        return SentimentFeatureSnapshot(
            timestamp=as_of,
            features=features,
            article_count=0,
            metadata={"phase": "3.1", "reason": "no_articles"},
        )

    # Filter to articles aligned at or before as_of
    filtered_news = filter_articles_by_as_of(news_events, as_of)

    # Validate no future data
    if validate:
        is_valid, errors = validate_no_future_data(filtered_news, as_of)
        if not is_valid:
            raise ValueError(f"Lookahead bias detected: {errors}")

    if filtered_news.empty:
        features = {name: None for name in get_phase_31_feature_names()}
        return SentimentFeatureSnapshot(
            timestamp=as_of,
            features=features,
            article_count=0,
            metadata={"phase": "3.1", "reason": "no_articles_before_as_of"},
        )

    # Compute article-level sentiment scores
    article_scores = _compute_article_scores(filtered_news, use_mock_inference)

    # Compute features
    features = _compute_features(article_scores, as_of)

    # Validate output
    if validate:
        is_valid, errors = validate_feature_snapshot(features, strict=True)
        if not is_valid:
            raise ValueError(f"Invalid feature snapshot: {errors}")

    # Build metadata
    metadata = {
        "phase": "3.1",
        "article_count": len(filtered_news),
        "mock_inference": use_mock_inference,
        "computed_features": get_phase_31_feature_names(),
    }

    return SentimentFeatureSnapshot(
        timestamp=as_of,
        features=features,
        article_count=len(filtered_news),
        metadata=metadata,
    )


def _compute_article_scores(
    news_events: pd.DataFrame,
    use_mock_inference: bool,
) -> pd.DataFrame:
    """
    Compute sentiment scores for each article.

    Args:
        news_events: Filtered news events
        use_mock_inference: Whether to use mock FinBERT

    Returns:
        DataFrame with columns: article_id, aligned_date, score, label
    """
    inference = FinBERTInference(use_mock=use_mock_inference)

    results = []
    for _, row in news_events.iterrows():
        # Combine headline and body for inference
        headline = row.get("headline", "")
        body = row.get("body") if pd.notna(row.get("body")) else ""
        text = f"{headline} {body}".strip()

        # Skip empty text
        if not text:
            continue

        # Infer sentiment
        result: SentimentResult = inference.infer(text)

        # Get aligned date
        ts_col = "aligned_timestamp" if "aligned_timestamp" in news_events.columns else "timestamp"
        aligned_ts = pd.Timestamp(row[ts_col])
        aligned_date = aligned_ts.date()

        results.append({
            "article_id": row.get("article_id"),
            "aligned_date": aligned_date,
            "score": result.score,
            "label": result.label,
            "confidence": result.confidence,
        })

    return pd.DataFrame(results)


def _compute_features(
    article_scores: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, float | None]:
    """
    Compute all sentiment features from article scores.

    Args:
        article_scores: DataFrame with article-level scores
        as_of: Reference timestamp

    Returns:
        Dict of feature_name -> value
    """
    features: dict[str, float | None] = {}

    # Get as_of date
    as_of_date = as_of.date()

    # Compute daily sentiment series for rolling calculations
    # Need 25 days for sentiment_volatility_20d + some buffer for news_sentiment_5d
    daily_sentiment = compute_daily_sentiment_series(
        article_scores,
        end_date=as_of_date,
        lookback_days=25,
    )

    # news_sentiment_daily: sentiment for as_of day
    if not daily_sentiment.empty and as_of_date in daily_sentiment.index:
        val = daily_sentiment.loc[as_of_date]
        features["news_sentiment_daily"] = None if pd.isna(val) else float(val)
    else:
        features["news_sentiment_daily"] = None

    # news_sentiment_5d: 5-day EWM
    features["news_sentiment_5d"] = compute_rolling_sentiment(
        daily_sentiment,
        window_days=5,
        use_ewm=True,
        ewm_span=5,
    )

    # sentiment_volatility_20d: 20-day std dev
    features["sentiment_volatility_20d"] = compute_sentiment_volatility(
        daily_sentiment,
        window_days=20,
    )

    return features
