"""
Sentiment label to numeric score mapping.

FinBERT outputs labels: positive, neutral, negative
These are mapped to numeric scores for aggregation.

From features.yaml:
- output_range: [-1.0, 1.0]
- Mapping: positive → +1, neutral → 0, negative → -1
"""

from typing import Literal

# Type alias for FinBERT sentiment labels
SentimentLabel = Literal["positive", "neutral", "negative"]

# Numeric mapping for sentiment labels
# This mapping is fixed and must not be changed without spec update
SENTIMENT_SCORE_MAP: dict[SentimentLabel, float] = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def label_to_score(label: SentimentLabel) -> float:
    """
    Convert FinBERT label to numeric score.

    Args:
        label: FinBERT output label (positive, neutral, negative)

    Returns:
        Numeric score in [-1.0, 1.0]

    Raises:
        ValueError: If label is not recognized

    Mathematical definition:
        positive → +1.0
        neutral  →  0.0
        negative → -1.0

    Economic intuition:
        Simple linear mapping preserves sentiment polarity.
        Neutral articles contribute zero to aggregation mean.
    """
    if label not in SENTIMENT_SCORE_MAP:
        raise ValueError(
            f"Unknown sentiment label: {label}. "
            f"Expected one of: {list(SENTIMENT_SCORE_MAP.keys())}"
        )
    return SENTIMENT_SCORE_MAP[label]


def score_to_label(score: float, threshold: float = 0.33) -> SentimentLabel:
    """
    Convert numeric score back to label (for interpretability).

    Args:
        score: Numeric sentiment score
        threshold: Threshold for positive/negative classification

    Returns:
        Sentiment label

    Note: This is primarily for debugging and explainability.
    The primary flow is label → score, not the reverse.
    """
    if score > threshold:
        return "positive"
    elif score < -threshold:
        return "negative"
    else:
        return "neutral"


def validate_score(score: float) -> bool:
    """
    Validate that a sentiment score is in valid range.

    Args:
        score: Score to validate

    Returns:
        True if score is in [-1.0, 1.0]
    """
    return -1.0 <= score <= 1.0
