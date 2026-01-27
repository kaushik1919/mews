"""
Sentiment feature validation against core-specs/features.yaml.

Ensures:
1. Feature names match spec exactly
2. No extra features
3. No missing required features
4. No NaN values (use None)
5. All data used has aligned_timestamp <= as_of

Fail fast on violations.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Path to core-specs
CORE_SPECS_DIR = Path(__file__).parent.parent.parent / "core-specs"
FEATURES_YAML = CORE_SPECS_DIR / "features.yaml"


def load_feature_spec() -> dict[str, Any]:
    """Load features.yaml specification."""
    if not FEATURES_YAML.exists():
        raise FileNotFoundError(
            f"Feature spec not found: {FEATURES_YAML}. "
            "Ensure core-specs/features.yaml exists."
        )

    with open(FEATURES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_sentiment_feature_names() -> list[str]:
    """
    Get list of all sentiment feature names from spec.

    Returns all sentiment features defined in features.yaml.
    """
    spec = load_feature_spec()
    sentiment = spec.get("sentiment", {})
    return list(sentiment.keys())


def get_phase_31_feature_names() -> list[str]:
    """
    Get list of sentiment features implemented in Phase 3.1.

    Phase 3.1 scope (from task specification):
    - news_sentiment_daily
    - news_sentiment_5d
    - sentiment_volatility_20d
    """
    return [
        "news_sentiment_daily",
        "news_sentiment_5d",
        "sentiment_volatility_20d",
    ]


def validate_feature_snapshot(
    features: dict[str, float | None],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate a sentiment feature snapshot against spec.

    Args:
        features: Dict of feature_name -> value (or None)
        strict: If True, require exact match to Phase 3.1 features

    Returns:
        Tuple of (is_valid, list of error messages)

    Validation rules:
    1. All feature names must exist in features.yaml
    2. Values must be float or None (no NaN)
    3. In strict mode, must have exactly Phase 3.1 features
    """
    import math

    errors: list[str] = []

    spec_features = get_sentiment_feature_names()
    phase_31_features = get_phase_31_feature_names()

    for name, value in features.items():
        # Check name is in spec
        if name not in spec_features:
            errors.append(f"Unknown feature: {name} (not in features.yaml sentiment)")

        # Check for NaN (should use None instead)
        if value is not None:
            if math.isnan(value):
                errors.append(f"Feature {name} has NaN value (use None instead)")

            # Check range for sentiment features
            if name in ["news_sentiment_daily", "news_sentiment_5d"]:
                if not -1.0 <= value <= 1.0:
                    errors.append(
                        f"Feature {name} out of range [-1, 1]: {value}"
                    )
            elif name == "sentiment_volatility_20d":
                if value < 0:
                    errors.append(
                        f"Feature {name} must be non-negative: {value}"
                    )

    if strict:
        # Check for missing Phase 3.1 features
        for name in phase_31_features:
            if name not in features:
                errors.append(f"Missing required Phase 3.1 feature: {name}")

        # Check for extra features
        for name in features.keys():
            if name not in phase_31_features:
                errors.append(f"Extra feature not in Phase 3.1 scope: {name}")

    return len(errors) == 0, errors


def validate_input_news_events(
    news_events: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Validate input news events DataFrame.

    Args:
        news_events: DataFrame from Phase 2.3 (financial_news)

    Returns:
        Tuple of (is_valid, list of error messages)

    Required columns:
    - article_id
    - aligned_timestamp (or timestamp)
    - headline
    - body (optional, can be null)
    """
    errors: list[str] = []

    if news_events is None:
        errors.append("news_events is None")
        return False, errors

    if not isinstance(news_events, pd.DataFrame):
        errors.append("news_events must be a pandas DataFrame")
        return False, errors

    if news_events.empty:
        # Empty is allowed, features will return None
        return True, []

    # Check required columns
    required_cols = ["article_id", "headline"]
    for col in required_cols:
        if col not in news_events.columns:
            errors.append(f"Missing required column: {col}")

    # Check for timestamp column (either name works)
    if "aligned_timestamp" not in news_events.columns and "timestamp" not in news_events.columns:
        errors.append("Missing timestamp column (need 'aligned_timestamp' or 'timestamp')")

    return len(errors) == 0, errors


def validate_no_future_data(
    news_events: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[bool, list[str]]:
    """
    Validate that no future data is being used.

    Args:
        news_events: Filtered news events DataFrame
        as_of: Reference timestamp

    Returns:
        Tuple of (is_valid, list of error messages)

    CRITICAL: This catches lookahead bias.
    """
    errors: list[str] = []

    if news_events is None or news_events.empty:
        return True, []

    # Determine timestamp column
    ts_col = "aligned_timestamp" if "aligned_timestamp" in news_events.columns else "timestamp"

    # Ensure timezone awareness
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")

    df_ts = news_events[ts_col]
    if df_ts.dt.tz is None:
        df_ts = df_ts.dt.tz_localize("UTC")

    # Check for any future timestamps
    future_mask = df_ts > as_of_ts
    future_count = future_mask.sum()

    if future_count > 0:
        errors.append(
            f"LOOKAHEAD BIAS: {future_count} articles have timestamp > as_of. "
            f"as_of={as_of_ts}, max_timestamp={df_ts.max()}"
        )

    return len(errors) == 0, errors


def get_feature_metadata(feature_name: str) -> dict[str, Any] | None:
    """
    Get metadata for a specific sentiment feature from spec.

    Args:
        feature_name: Name of feature

    Returns:
        Feature specification dict, or None if not found
    """
    spec = load_feature_spec()
    sentiment = spec.get("sentiment", {})

    if feature_name in sentiment:
        return sentiment[feature_name]

    return None
