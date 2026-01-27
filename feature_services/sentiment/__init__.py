"""
MEWS-FIN Sentiment Feature Service.

Computes sentiment-based systemic risk features from financial news.
Uses frozen FinBERT model for deterministic inference.

Phase 3.1 scope (from features.yaml):
- news_sentiment_daily: Daily aggregate sentiment
- news_sentiment_5d: 5-day rolling sentiment
- sentiment_volatility_20d: 20-day sentiment volatility

NOT in scope:
- Topic modeling
- Entity-level sentiment
- News volume features
- Any model training or fine-tuning
"""

from .service import SentimentFeatureSnapshot, compute_sentiment_features

__all__ = ["compute_sentiment_features", "SentimentFeatureSnapshot"]
