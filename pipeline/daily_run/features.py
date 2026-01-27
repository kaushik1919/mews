"""
Feature Computation Pipeline Step.

MEWS-FIN Phase 5A: Compute all features from ingested data.

Calls Phase 3 feature services and aggregates results.
Mock mode generates plausible feature values.
"""

from __future__ import annotations

import numpy as np

from .context import PipelineContext


def run_features(ctx: PipelineContext) -> None:
    """
    Execute feature computation step.

    Computes numeric, sentiment, and graph features from ingested data.
    In mock mode, generates synthetic features.

    Args:
        ctx: Pipeline context (modified in place)
    """
    timing = ctx.start_step("features")

    try:
        if ctx.config.mock_mode:
            _run_mock_features(ctx)
        else:
            _run_live_features(ctx)

        ctx.complete_step(timing, success=True)

    except Exception as e:
        ctx.complete_step(timing, success=False, error=str(e))
        raise


def _run_live_features(ctx: PipelineContext) -> None:
    """
    Compute features from actual ingested data.

    Calls the Phase 3 feature services.
    """
    # Check we have required datasets
    if not ctx.ingestion.is_complete:
        raise ValueError("Ingestion incomplete - cannot compute features")

    # Compute numeric features
    if "numeric" in ctx.config.features.services:
        _compute_numeric_features(ctx)

    # Compute sentiment features
    if "sentiment" in ctx.config.features.services:
        _compute_sentiment_features(ctx)

    # Compute graph features
    if "graph" in ctx.config.features.services:
        _compute_graph_features(ctx)


def _compute_numeric_features(ctx: PipelineContext) -> None:
    """Compute numeric features from market data."""
    try:
        from feature_services.numeric import compute_numeric_features

        datasets = {
            "market_prices": ctx.ingestion.datasets.get("market_prices"),
            "volatility_indices": ctx.ingestion.datasets.get("volatility_indices"),
        }

        snapshot = compute_numeric_features(
            datasets=datasets,
            as_of=ctx.as_of,
        )

        ctx.features.numeric_features = snapshot.features
        ctx.features.metadata["numeric_complete"] = snapshot.is_complete

    except Exception as e:
        ctx.add_warning(f"Numeric features failed: {e}")
        _fill_mock_numeric_features(ctx)


def _compute_sentiment_features(ctx: PipelineContext) -> None:
    """Compute sentiment features from news data."""
    try:
        from feature_services.sentiment import compute_sentiment_features

        news_df = ctx.ingestion.datasets.get("financial_news")

        snapshot = compute_sentiment_features(
            news_events=news_df,
            as_of=ctx.as_of,
        )

        ctx.features.sentiment_features = snapshot.features
        ctx.features.metadata["sentiment_complete"] = snapshot.is_complete

    except Exception as e:
        ctx.add_warning(f"Sentiment features failed: {e}")
        _fill_mock_sentiment_features(ctx)


def _compute_graph_features(ctx: PipelineContext) -> None:
    """Compute graph features from market data."""
    try:
        from feature_services.graph import compute_graph_features

        market_prices = ctx.ingestion.datasets.get("market_prices")

        snapshot = compute_graph_features(
            market_prices=market_prices,
            as_of=ctx.as_of,
        )

        ctx.features.graph_features = snapshot.features
        ctx.features.metadata["graph_complete"] = snapshot.is_complete

    except Exception as e:
        ctx.add_warning(f"Graph features failed: {e}")
        _fill_mock_graph_features(ctx)


def _run_mock_features(ctx: PipelineContext) -> None:
    """Generate mock feature values for testing."""
    _fill_mock_numeric_features(ctx)
    _fill_mock_sentiment_features(ctx)
    _fill_mock_graph_features(ctx)
    ctx.features.metadata["mode"] = "mock"


def _fill_mock_numeric_features(ctx: PipelineContext) -> None:
    """Generate mock numeric features."""
    np.random.seed(46)

    ctx.features.numeric_features = {
        "realized_volatility_20d": np.random.uniform(0.10, 0.35),
        "realized_volatility_60d": np.random.uniform(0.12, 0.30),
        "volatility_ratio": np.random.uniform(0.8, 1.4),
        "vix_level": np.random.uniform(14, 35),
        "max_drawdown_20d": np.random.uniform(-0.08, -0.02),
        "max_drawdown_60d": np.random.uniform(-0.15, -0.05),
        "volume_zscore_20d": np.random.uniform(-1.5, 2.5),
        "volume_price_divergence": np.random.uniform(-0.5, 0.5),
    }


def _fill_mock_sentiment_features(ctx: PipelineContext) -> None:
    """Generate mock sentiment features."""
    np.random.seed(47)

    ctx.features.sentiment_features = {
        "news_sentiment_daily": np.random.uniform(-0.4, 0.3),
        "news_sentiment_5d": np.random.uniform(-0.3, 0.2),
        "sentiment_volatility_20d": np.random.uniform(0.2, 0.6),
    }


def _fill_mock_graph_features(ctx: PipelineContext) -> None:
    """Generate mock graph features."""
    np.random.seed(48)

    ctx.features.graph_features = {
        "avg_pairwise_correlation_20d": np.random.uniform(0.35, 0.75),
        "correlation_dispersion_20d": np.random.uniform(0.08, 0.25),
        "max_sector_correlation_20d": np.random.uniform(0.50, 0.85),
        "degree_centrality_spy": np.random.uniform(0.4, 0.8),
        "centrality_shift_20d": np.random.uniform(-0.15, 0.15),
    }
