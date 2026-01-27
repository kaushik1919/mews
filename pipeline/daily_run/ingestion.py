"""
Ingestion Pipeline Step.

MEWS-FIN Phase 5A: Load and prepare data from all sources.

Calls Phase 2 data ingestion adapters and produces aligned datasets.
Mock mode generates synthetic data for testing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .context import PipelineContext


def run_ingestion(ctx: PipelineContext) -> None:
    """
    Execute data ingestion step.

    Loads data from all configured sources and stores in context.
    In mock mode, generates synthetic data.

    Args:
        ctx: Pipeline context (modified in place)
    """
    timing = ctx.start_step("ingestion")

    try:
        if ctx.config.mock_mode:
            _run_mock_ingestion(ctx)
        else:
            _run_live_ingestion(ctx)

        ctx.complete_step(timing, success=True)

    except Exception as e:
        ctx.complete_step(timing, success=False, error=str(e))
        raise


def _run_live_ingestion(ctx: PipelineContext) -> None:
    """
    Run live data ingestion from actual sources.

    This would call the actual Phase 2 adapters in production.
    """
    # TODO: Wire to actual adapters when live data is available
    # For now, fail fast with clear message
    ctx.add_warning("Live ingestion not implemented - falling back to mock mode")
    _run_mock_ingestion(ctx)


def _run_mock_ingestion(ctx: PipelineContext) -> None:
    """
    Generate mock data for testing and demos.

    Creates realistic-looking synthetic data for all sources.
    """
    run_date = ctx.run_date

    # Generate market prices
    ctx.ingestion.datasets["market_prices"] = _generate_mock_market_prices(run_date)
    ctx.ingestion.sources_loaded.append("market_prices")

    # Generate volatility indices
    ctx.ingestion.datasets["volatility_indices"] = _generate_mock_volatility(run_date)
    ctx.ingestion.sources_loaded.append("volatility_indices")

    # Generate macro rates
    ctx.ingestion.datasets["macro_rates"] = _generate_mock_macro_rates(run_date)
    ctx.ingestion.sources_loaded.append("macro_rates")

    # Generate financial news
    ctx.ingestion.datasets["financial_news"] = _generate_mock_news(run_date)
    ctx.ingestion.sources_loaded.append("financial_news")

    ctx.ingestion.metadata["mode"] = "mock"
    ctx.ingestion.metadata["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()


def _generate_mock_market_prices(run_date: date, days: int = 90) -> pd.DataFrame:
    """Generate mock market price data."""
    np.random.seed(42)  # Deterministic for testing

    dates = pd.date_range(
        end=pd.Timestamp(run_date),
        periods=days,
        freq="B",  # Business days
        tz="UTC",
    )

    assets = ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD", "XLF", "XLE"]

    data = []
    for asset in assets:
        base_price = np.random.uniform(100, 400)
        returns = np.random.normal(0.0003, 0.012, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        volumes = np.random.uniform(1e6, 1e8, len(dates))

        for i, dt in enumerate(dates):
            data.append({
                "timestamp": dt,
                "asset_id": asset,
                "close": prices[i],
                "volume": volumes[i],
                "high": prices[i] * (1 + abs(np.random.normal(0, 0.005))),
                "low": prices[i] * (1 - abs(np.random.normal(0, 0.005))),
            })

    return pd.DataFrame(data)


def _generate_mock_volatility(run_date: date, days: int = 90) -> pd.DataFrame:
    """Generate mock volatility index data."""
    np.random.seed(43)

    dates = pd.date_range(
        end=pd.Timestamp(run_date),
        periods=days,
        freq="B",
        tz="UTC",
    )

    # VIX-like mean-reverting process
    vix_values = [18.0]
    for _ in range(len(dates) - 1):
        # Mean reversion to 18, with occasional spikes
        mean_reversion = 0.1 * (18.0 - vix_values[-1])
        shock = np.random.normal(0, 1.5)
        new_vix = max(10, vix_values[-1] + mean_reversion + shock)
        vix_values.append(new_vix)

    data = []
    for i, dt in enumerate(dates):
        data.append({
            "timestamp": dt,
            "index_id": "VIX",
            "close": vix_values[i],
            "open": vix_values[i] * (1 + np.random.normal(0, 0.02)),
            "high": vix_values[i] * (1 + abs(np.random.normal(0, 0.05))),
            "low": vix_values[i] * (1 - abs(np.random.normal(0, 0.05))),
        })

    return pd.DataFrame(data)


def _generate_mock_macro_rates(run_date: date, days: int = 90) -> pd.DataFrame:
    """Generate mock macro rates data."""
    np.random.seed(44)

    dates = pd.date_range(
        end=pd.Timestamp(run_date),
        periods=days,
        freq="B",
        tz="UTC",
    )

    # Treasury rates
    rates = {
        "US10Y": 4.2,
        "US2Y": 4.8,
        "US3M": 5.3,
    }

    data = []
    for dt in dates:
        for rate_id, base_rate in rates.items():
            value = base_rate + np.random.normal(0, 0.02)
            data.append({
                "timestamp": dt,
                "rate_id": rate_id,
                "value": value,
            })

    return pd.DataFrame(data)


def _generate_mock_news(run_date: date, days: int = 30) -> pd.DataFrame:
    """Generate mock financial news data."""
    np.random.seed(45)

    data = []
    start_date = run_date - timedelta(days=days)

    for day_offset in range(days + 1):
        current_date = start_date + timedelta(days=day_offset)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # 3-8 articles per day
        n_articles = np.random.randint(3, 9)
        for i in range(n_articles):
            hour = np.random.randint(6, 20)
            minute = np.random.randint(0, 60)

            # Sentiment: slightly negative bias for realism
            sentiment = np.random.normal(-0.05, 0.35)
            sentiment = max(-1.0, min(1.0, sentiment))

            data.append({
                "timestamp": pd.Timestamp(
                    year=current_date.year,
                    month=current_date.month,
                    day=current_date.day,
                    hour=hour,
                    minute=minute,
                    tz="UTC",
                ),
                "source": np.random.choice(["reuters", "bloomberg", "wsj", "ft"]),
                "headline": f"Mock headline {day_offset}_{i}",
                "sentiment_score": sentiment,
                "relevance_score": np.random.uniform(0.5, 1.0),
            })

    return pd.DataFrame(data)
