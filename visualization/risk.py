"""
Risk Engine Visualization.

MEWS Phase 5B: Plots for heuristic, ML, and ensemble risk engines.

Generates:
    - heuristic_risk_score.png
    - ml_vs_heuristic_comparison.png
    - calibration_curve.png
    - ensemble_vs_components.png
    - shap_global_importance.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COLORS,
    RISK_ENGINE_DIR,
    create_figure,
    create_subplots,
    save_figure,
    shade_crisis_periods,
)

# Crisis periods for shading
CRISIS_PERIODS = [
    ("2008-09-15", "2009-03-09"),
    ("2011-07-01", "2011-10-04"),
    ("2020-02-20", "2020-03-23"),
    ("2022-01-03", "2022-10-12"),
]


def generate_mock_risk_scores(
    start: str = "2007-01-01",
    end: str = "2023-12-31",
    seed: int = 200,
) -> pd.DataFrame:
    """Generate mock risk score time series."""
    np.random.seed(seed)

    dates = pd.date_range(start, end, freq="B")
    n = len(dates)

    # Base heuristic score
    heuristic = 0.35 + 0.10 * np.sin(np.linspace(0, 8 * np.pi, n))
    heuristic += np.random.normal(0, 0.03, n)

    # Add crisis spikes
    for crisis_start, crisis_end in CRISIS_PERIODS:
        mask = (dates >= crisis_start) & (dates <= crisis_end)
        heuristic[mask] += np.random.uniform(0.25, 0.45, mask.sum())

    heuristic = np.clip(heuristic, 0.05, 0.95)

    # ML score (slightly more reactive)
    ml = heuristic + np.random.normal(0.02, 0.05, n)
    ml = np.clip(ml, 0.05, 0.95)

    # Ensemble (smoothed combination)
    ensemble = 0.35 * heuristic + 0.65 * ml
    # Apply smoothing
    ensemble = pd.Series(ensemble).ewm(span=3).mean().values
    ensemble = np.clip(ensemble, 0.05, 0.95)

    return pd.DataFrame({
        "date": dates,
        "heuristic": heuristic,
        "ml": ml,
        "ensemble": ensemble,
    }).set_index("date")


def plot_heuristic_risk_score(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate heuristic risk score time series plot.

    Args:
        df: DataFrame with heuristic column
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_risk_scores()

    if output_path is None:
        output_path = RISK_ENGINE_DIR / "heuristic_risk_score.png"

    fig, ax = create_figure("wide")

    shade_crisis_periods(ax, CRISIS_PERIODS)

    ax.plot(
        df.index, df["heuristic"],
        color=COLORS["heuristic"],
        linewidth=1.2,
        label="Heuristic Risk Score",
    )

    # Add regime threshold lines
    ax.axhline(y=0.25, color=COLORS["moderate_risk"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=0.50, color=COLORS["high_risk"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=0.75, color=COLORS["extreme_risk"], linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Date")
    ax.set_title("Heuristic Risk Score Over Time")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)

    save_figure(fig, output_path)

    return output_path


def plot_ml_vs_heuristic(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate ML vs heuristic comparison plot.

    Args:
        df: DataFrame with heuristic and ml columns
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_risk_scores()

    if output_path is None:
        output_path = RISK_ENGINE_DIR / "ml_vs_heuristic_comparison.png"

    fig, axes = create_subplots(2, 1, "double")

    # Time series comparison
    ax1 = axes[0]
    shade_crisis_periods(ax1, CRISIS_PERIODS)

    ax1.plot(
        df.index, df["heuristic"],
        color=COLORS["heuristic"],
        linewidth=1.0,
        alpha=0.8,
        label="Heuristic",
    )
    ax1.plot(
        df.index, df["ml"],
        color=COLORS["ml"],
        linewidth=1.0,
        alpha=0.8,
        label="ML (Random Forest)",
    )

    ax1.set_ylabel("Risk Score")
    ax1.set_title("Heuristic vs ML Risk Scores")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 1.0)

    # Scatter plot
    ax2 = axes[1]

    ax2.scatter(
        df["heuristic"], df["ml"],
        alpha=0.3,
        s=5,
        color=COLORS["primary"],
    )

    # Diagonal line (perfect agreement)
    ax2.plot([0, 1], [0, 1], color=COLORS["secondary"], linestyle="--", linewidth=1.0)

    ax2.set_xlabel("Heuristic Score")
    ax2.set_ylabel("ML Score")
    ax2.set_title("Score Agreement")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, output_path)

    return output_path


def plot_calibration_curve(
    output_path: Path | None = None,
) -> Path:
    """
    Generate calibration curve plot.

    Shows how raw ML scores map to calibrated probabilities.

    Args:
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = RISK_ENGINE_DIR / "calibration_curve.png"

    np.random.seed(201)

    fig, ax = create_figure("single")

    # Generate calibration data
    n_bins = 10
    raw_scores = np.linspace(0.05, 0.95, n_bins)

    # Simulate uncalibrated (overconfident) predictions
    uncalibrated = raw_scores ** 0.7  # Overconfident

    # Simulated calibrated predictions (closer to diagonal)
    calibrated = raw_scores + np.random.normal(0, 0.02, n_bins)
    calibrated = np.clip(calibrated, 0.05, 0.95)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color=COLORS["secondary"], linestyle="--", linewidth=1.0, label="Perfect Calibration")

    # Uncalibrated
    ax.plot(
        raw_scores, uncalibrated,
        color=COLORS["high_risk"],
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Before Calibration",
    )

    # Calibrated
    ax.plot(
        raw_scores, calibrated,
        color=COLORS["ml"],
        linewidth=2.0,
        marker="s",
        markersize=6,
        label="After Calibration",
    )

    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    save_figure(fig, output_path)

    return output_path


def plot_ensemble_vs_components(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate ensemble vs component models plot.

    Args:
        df: DataFrame with all score columns
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if df is None:
        df = generate_mock_risk_scores()

    if output_path is None:
        output_path = RISK_ENGINE_DIR / "ensemble_vs_components.png"

    fig, ax = create_figure("wide")

    shade_crisis_periods(ax, CRISIS_PERIODS)

    # Component scores (lighter)
    ax.plot(
        df.index, df["heuristic"],
        color=COLORS["heuristic"],
        linewidth=0.8,
        alpha=0.5,
        label="Heuristic (35%)",
    )
    ax.plot(
        df.index, df["ml"],
        color=COLORS["ml"],
        linewidth=0.8,
        alpha=0.5,
        label="ML (65%)",
    )

    # Ensemble (prominent)
    ax.plot(
        df.index, df["ensemble"],
        color=COLORS["ensemble"],
        linewidth=2.0,
        label="Ensemble (calibrated)",
    )

    # Regime thresholds
    ax.axhline(y=0.50, color=COLORS["high_risk"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=0.75, color=COLORS["extreme_risk"], linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Date")
    ax.set_title("Ensemble Risk Score and Component Contributions")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.0)

    save_figure(fig, output_path)

    return output_path


def plot_shap_importance(
    output_path: Path | None = None,
) -> Path:
    """
    Generate SHAP global feature importance plot.

    Args:
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = RISK_ENGINE_DIR / "shap_global_importance.png"

    np.random.seed(202)

    fig, ax = create_figure("single")

    # Mock SHAP importance values
    features = [
        "realized_volatility_20d",
        "avg_pairwise_correlation_20d",
        "max_drawdown_60d",
        "vix_level",
        "news_sentiment_5d",
        "volume_zscore_20d",
        "correlation_dispersion_20d",
        "sentiment_volatility_20d",
    ]

    importance = np.array([0.28, 0.22, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03])

    # Sort by importance
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = importance[sorted_idx]

    # Clean feature names
    clean_names = [f.replace("_", " ").replace("20d", "(20d)").replace("60d", "(60d)").replace("5d", "(5d)") for f in features]

    colors = [COLORS["correlation"] if "correlation" in f else
              COLORS["volatility"] if "volatility" in f or "vix" in f else
              COLORS["drawdown"] if "drawdown" in f else
              COLORS["sentiment"] if "sentiment" in f else
              COLORS["liquidity"] for f in features]

    ax.barh(clean_names, importance, color=colors, alpha=0.8)

    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.set_xlim(0, max(importance) * 1.1)

    save_figure(fig, output_path)

    return output_path


def generate_all_risk_plots(mock: bool = True) -> list[Path]:
    """
    Generate all risk engine plots.

    Args:
        mock: If True, use mock data

    Returns:
        List of paths to generated figures
    """
    df = generate_mock_risk_scores() if mock else None

    paths = [
        plot_heuristic_risk_score(df),
        plot_ml_vs_heuristic(df),
        plot_calibration_curve(),
        plot_ensemble_vs_components(df),
        plot_shap_importance(),
    ]

    return paths
