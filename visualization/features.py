"""
Feature Visualization.

MEWS Phase 5B: Plots for feature-level dynamics.

Generates:
    - volatility_and_drawdown_timeseries.png
    - news_sentiment_vs_market.png
    - avg_correlation_timeseries.png
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    COLORS,
    FEATURES_DIR,
    create_figure,
    create_subplots,
    save_figure,
    shade_crisis_periods,
)


# Crisis periods for shading
CRISIS_PERIODS = [
    ("2008-09-15", "2009-03-09"),  # GFC
    ("2011-07-01", "2011-10-04"),  # Eurozone
    ("2020-02-20", "2020-03-23"),  # COVID
    ("2022-01-03", "2022-10-12"),  # Rate hikes
]


def generate_mock_timeseries(
    start: str = "2007-01-01",
    end: str = "2023-12-31",
    seed: int = 100,
) -> pd.DataFrame:
    """Generate mock feature time series."""
    np.random.seed(seed)
    
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    
    # Base volatility with regime changes
    base_vol = 0.15 + 0.05 * np.sin(np.linspace(0, 8 * np.pi, n))
    
    # Add crisis spikes
    vol = base_vol.copy()
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        vol[mask] += np.random.uniform(0.10, 0.25, mask.sum())
    
    vol += np.random.normal(0, 0.02, n)
    vol = np.clip(vol, 0.05, 0.60)
    
    # Drawdown (negative values)
    drawdown = -np.abs(np.random.normal(0.05, 0.03, n))
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        drawdown[mask] -= np.random.uniform(0.10, 0.30, mask.sum())
    drawdown = np.clip(drawdown, -0.55, 0)
    
    # Correlation
    correlation = 0.35 + 0.15 * np.sin(np.linspace(0, 6 * np.pi, n))
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        correlation[mask] += np.random.uniform(0.20, 0.40, mask.sum())
    correlation += np.random.normal(0, 0.03, n)
    correlation = np.clip(correlation, 0.10, 0.95)
    
    # Sentiment
    sentiment = np.random.normal(0.0, 0.15, n)
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        sentiment[mask] -= np.random.uniform(0.20, 0.50, mask.sum())
    sentiment = np.clip(sentiment, -1.0, 1.0)
    
    # Market returns
    returns = np.random.normal(0.0003, 0.012, n)
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        returns[mask] -= np.random.uniform(0.01, 0.05, mask.sum())
    
    return pd.DataFrame({
        "date": dates,
        "realized_volatility_20d": vol,
        "max_drawdown_60d": drawdown,
        "avg_pairwise_correlation_20d": correlation,
        "news_sentiment_5d": sentiment,
        "market_return": returns,
        "cumulative_return": (1 + returns).cumprod(),
    }).set_index("date")


def plot_volatility_and_drawdown(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate volatility and drawdown time series plot.
    
    Args:
        df: DataFrame with volatility and drawdown columns
        output_path: Output path (uses default if None)
        
    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_timeseries()
    
    if output_path is None:
        output_path = FEATURES_DIR / "volatility_and_drawdown_timeseries.png"
    
    fig, axes = create_subplots(2, 1, "double")
    
    # Volatility panel
    ax1 = axes[0]
    shade_crisis_periods(ax1, CRISIS_PERIODS)
    ax1.plot(
        df.index, df["realized_volatility_20d"],
        color=COLORS["volatility"],
        linewidth=1.2,
        label="20-day Realized Volatility",
    )
    ax1.axhline(y=0.20, color=COLORS["secondary"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Volatility")
    ax1.set_title("Realized Volatility Over Time")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 0.65)
    
    # Drawdown panel
    ax2 = axes[1]
    shade_crisis_periods(ax2, CRISIS_PERIODS)
    ax2.fill_between(
        df.index, 0, df["max_drawdown_60d"],
        color=COLORS["drawdown"],
        alpha=0.5,
        label="60-day Max Drawdown",
    )
    ax2.plot(
        df.index, df["max_drawdown_60d"],
        color=COLORS["drawdown"],
        linewidth=1.0,
    )
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Maximum Drawdown Over Time")
    ax2.legend(loc="lower left")
    ax2.set_ylim(-0.60, 0.05)
    
    fig.tight_layout()
    save_figure(fig, output_path)
    
    return output_path


def plot_sentiment_vs_market(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate sentiment vs market returns plot.
    
    Args:
        df: DataFrame with sentiment and market columns
        output_path: Output path (uses default if None)
        
    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_timeseries()
    
    if output_path is None:
        output_path = FEATURES_DIR / "news_sentiment_vs_market.png"
    
    fig, ax1 = create_figure("wide")
    
    shade_crisis_periods(ax1, CRISIS_PERIODS)
    
    # Sentiment on primary axis
    color1 = COLORS["sentiment"]
    ax1.plot(
        df.index, df["news_sentiment_5d"],
        color=color1,
        linewidth=1.2,
        label="5-day News Sentiment",
    )
    ax1.axhline(y=0, color=COLORS["secondary"], linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Sentiment Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-1.0, 1.0)
    
    # Cumulative return on secondary axis
    ax2 = ax1.twinx()
    color2 = COLORS["primary"]
    ax2.plot(
        df.index, df["cumulative_return"],
        color=color2,
        linewidth=1.0,
        alpha=0.7,
        label="Cumulative Market Return",
    )
    ax2.set_ylabel("Cumulative Return", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    ax1.set_xlabel("Date")
    ax1.set_title("News Sentiment and Market Performance")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    save_figure(fig, output_path)
    
    return output_path


def plot_correlation_timeseries(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate average correlation time series plot.
    
    Args:
        df: DataFrame with correlation column
        output_path: Output path (uses default if None)
        
    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_timeseries()
    
    if output_path is None:
        output_path = FEATURES_DIR / "avg_correlation_timeseries.png"
    
    fig, ax = create_figure("wide")
    
    shade_crisis_periods(ax, CRISIS_PERIODS)
    
    ax.plot(
        df.index, df["avg_pairwise_correlation_20d"],
        color=COLORS["correlation"],
        linewidth=1.2,
        label="20-day Avg Pairwise Correlation",
    )
    
    # Add threshold lines
    ax.axhline(y=0.50, color=COLORS["moderate_risk"], linestyle="--", linewidth=0.8, alpha=0.7, label="Elevated (0.50)")
    ax.axhline(y=0.70, color=COLORS["high_risk"], linestyle="--", linewidth=0.8, alpha=0.7, label="High (0.70)")
    
    ax.set_ylabel("Average Pairwise Correlation")
    ax.set_xlabel("Date")
    ax.set_title("Cross-Asset Correlation Dynamics")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)
    
    save_figure(fig, output_path)
    
    return output_path


def generate_all_feature_plots(mock: bool = True) -> list[Path]:
    """
    Generate all feature plots.
    
    Args:
        mock: If True, use mock data
        
    Returns:
        List of paths to generated figures
    """
    df = generate_mock_timeseries() if mock else None
    
    paths = [
        plot_volatility_and_drawdown(df),
        plot_sentiment_vs_market(df),
        plot_correlation_timeseries(df),
    ]
    
    return paths
