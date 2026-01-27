"""
Publication lag rules.

Defines publication lag rules from core-specs/time_alignment.yaml.
These rules determine when data becomes "available" for use.

From time_alignment.yaml:
- market_prices: lag = 0d (available at market close)
- macro_rates: lag = 1d (FRED publishes next business day)
- financial_news: lag = 0d (available at publication)
- sentiment_scores: lag = 0d (computed on ingestion)
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum


class DatasetType(Enum):
    """Supported dataset types."""
    MARKET_PRICES = "market_prices"
    VOLATILITY_INDICES = "volatility_indices"
    MACRO_RATES = "macro_rates"
    FINANCIAL_NEWS = "financial_news"
    SENTIMENT = "sentiment"


@dataclass(frozen=True)
class LagRule:
    """
    Publication lag rule for a dataset type.

    Attributes:
        dataset: The dataset type
        lag_days: Number of days lag (0 = same day)
        alignment_mode: How to align timestamps
        description: Human-readable explanation
    """
    dataset: DatasetType
    lag_days: int
    alignment_mode: str
    description: str


# Lag rules from core-specs/time_alignment.yaml
# These are immutable and must match the spec exactly
LAG_RULES: dict[DatasetType, LagRule] = {
    DatasetType.MARKET_PRICES: LagRule(
        dataset=DatasetType.MARKET_PRICES,
        lag_days=0,
        alignment_mode="native",
        description="Available at market close",
    ),
    DatasetType.VOLATILITY_INDICES: LagRule(
        dataset=DatasetType.VOLATILITY_INDICES,
        lag_days=0,
        alignment_mode="native",
        description="VIX closes with NYSE",
    ),
    DatasetType.MACRO_RATES: LagRule(
        dataset=DatasetType.MACRO_RATES,
        lag_days=1,
        alignment_mode="assign_to_previous_close",
        description="FRED typically publishes next business day",
    ),
    DatasetType.FINANCIAL_NEWS: LagRule(
        dataset=DatasetType.FINANCIAL_NEWS,
        lag_days=0,
        alignment_mode="assign_to_same_day_close",
        description="Available at publication time",
    ),
    DatasetType.SENTIMENT: LagRule(
        dataset=DatasetType.SENTIMENT,
        lag_days=0,
        alignment_mode="same_as_news",
        description="Computed on article ingestion",
    ),
}


def get_publication_lag(dataset_type: DatasetType) -> LagRule:
    """
    Get the publication lag rule for a dataset type.

    Args:
        dataset_type: The dataset type

    Returns:
        LagRule for the dataset

    Raises:
        KeyError: If dataset type is not defined
    """
    if dataset_type not in LAG_RULES:
        raise KeyError(
            f"No lag rule defined for dataset type: {dataset_type}. "
            f"Known types: {list(LAG_RULES.keys())}"
        )
    return LAG_RULES[dataset_type]


def get_lag_timedelta(dataset_type: DatasetType) -> timedelta:
    """
    Get the publication lag as a timedelta.

    Args:
        dataset_type: The dataset type

    Returns:
        timedelta representing the lag
    """
    rule = get_publication_lag(dataset_type)
    return timedelta(days=rule.lag_days)
