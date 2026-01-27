"""
Evaluation Visualization.

MEWS Phase 5B: Plots for lead time, false positive, and comparison analysis.

Generates:
    - lead_time_bar_chart.png
    - false_positive_rate_by_threshold.png
    - false_alarm_duration.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COLORS,
    EVALUATION_DIR,
    create_figure,
    save_figure,
)

# Crisis events for evaluation
CRISIS_EVENTS = ["GFC 2008", "Eurozone 2011", "COVID 2020", "Rate Hikes 2022"]


def generate_mock_lead_times() -> dict[str, dict[str, int]]:
    """Generate mock lead time data (days before crisis peak)."""
    np.random.seed(300)

    return {
        "GFC 2008": {
            "heuristic": 18,
            "ml": 22,
            "ensemble": 25,
        },
        "Eurozone 2011": {
            "heuristic": 12,
            "ml": 15,
            "ensemble": 17,
        },
        "COVID 2020": {
            "heuristic": 8,
            "ml": 10,
            "ensemble": 11,
        },
        "Rate Hikes 2022": {
            "heuristic": 14,
            "ml": 18,
            "ensemble": 21,
        },
    }


def generate_mock_false_positive_data() -> pd.DataFrame:
    """Generate mock false positive rate data by threshold."""
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    heuristic_fpr = [0.25, 0.18, 0.12, 0.08, 0.04, 0.01]
    ml_fpr = [0.22, 0.15, 0.09, 0.05, 0.02, 0.01]
    ensemble_fpr = [0.19, 0.12, 0.07, 0.04, 0.02, 0.01]

    return pd.DataFrame({
        "threshold": thresholds,
        "heuristic": heuristic_fpr,
        "ml": ml_fpr,
        "ensemble": ensemble_fpr,
    })


def generate_mock_alarm_durations() -> dict[str, list[int]]:
    """Generate mock false alarm duration data."""
    np.random.seed(301)

    return {
        "heuristic": [3, 5, 2, 8, 4, 6, 3, 7, 2, 5, 9, 4],
        "ml": [2, 4, 3, 5, 2, 4, 2, 6, 3, 4, 5, 3],
        "ensemble": [2, 3, 2, 4, 2, 3, 2, 4, 2, 3, 4, 2],
    }


def plot_lead_time_bar_chart(
    lead_times: dict[str, dict[str, int]] | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate lead time comparison bar chart.

    Shows days of advance warning before each crisis.

    Args:
        lead_times: Dict mapping crisis -> model -> days
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if lead_times is None:
        lead_times = generate_mock_lead_times()

    if output_path is None:
        output_path = EVALUATION_DIR / "lead_time_bar_chart.png"

    fig, ax = create_figure("wide")

    crises = list(lead_times.keys())
    models = ["heuristic", "ml", "ensemble"]
    x = np.arange(len(crises))
    width = 0.25

    model_colors = {
        "heuristic": COLORS["heuristic"],
        "ml": COLORS["ml"],
        "ensemble": COLORS["ensemble"],
    }

    for i, model in enumerate(models):
        values = [lead_times[crisis][model] for crisis in crises]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model.capitalize(), color=model_colors[model], alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Lead Time (Business Days)")
    ax.set_xlabel("Crisis Event")
    ax.set_title("Advance Warning Before Crisis Peak (Threshold = 0.60)")
    ax.set_xticks(x)
    ax.set_xticklabels(crises, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(lt.values()) for lt in lead_times.values()) * 1.2)

    save_figure(fig, output_path)

    return output_path


def plot_false_positive_rate(
    fpr_data: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate false positive rate by threshold plot.

    Args:
        fpr_data: DataFrame with threshold and model columns
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if fpr_data is None:
        fpr_data = generate_mock_false_positive_data()

    if output_path is None:
        output_path = EVALUATION_DIR / "false_positive_rate_by_threshold.png"

    fig, ax = create_figure("single")

    ax.plot(
        fpr_data["threshold"], fpr_data["heuristic"],
        color=COLORS["heuristic"],
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Heuristic",
    )
    ax.plot(
        fpr_data["threshold"], fpr_data["ml"],
        color=COLORS["ml"],
        linewidth=2.0,
        marker="s",
        markersize=6,
        label="ML",
    )
    ax.plot(
        fpr_data["threshold"], fpr_data["ensemble"],
        color=COLORS["ensemble"],
        linewidth=2.0,
        marker="^",
        markersize=6,
        label="Ensemble",
    )

    ax.set_xlabel("Alert Threshold")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate by Alert Threshold")
    ax.legend(loc="upper right")
    ax.set_xlim(0.35, 0.95)
    ax.set_ylim(0, max(fpr_data["heuristic"]) * 1.15)

    ax.axvline(x=0.60, color=COLORS["secondary"], linestyle="--", linewidth=1.0, alpha=0.6, label="Default Threshold")

    save_figure(fig, output_path)

    return output_path


def plot_false_alarm_duration(
    durations: dict[str, list[int]] | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Generate false alarm duration distribution plot.

    Args:
        durations: Dict mapping model -> list of durations
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if durations is None:
        durations = generate_mock_alarm_durations()

    if output_path is None:
        output_path = EVALUATION_DIR / "false_alarm_duration.png"

    fig, ax = create_figure("single")

    models = ["heuristic", "ml", "ensemble"]
    positions = [1, 2, 3]

    # Box plots
    bp_data = [durations[m] for m in models]
    bp = ax.boxplot(
        bp_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )

    # Color boxes
    colors = [COLORS["heuristic"], COLORS["ml"], COLORS["ensemble"]]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add scatter points
    for pos, model in zip(positions, models, strict=True):
        y = durations[model]
        x = np.random.normal(pos, 0.04, len(y))
        ax.scatter(x, y, alpha=0.4, s=20, color=COLORS["primary"])

    ax.set_ylabel("Duration (Business Days)")
    ax.set_xlabel("Model")
    ax.set_title("False Alarm Streak Duration")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Heuristic", "ML", "Ensemble"])

    # Add mean annotations
    for pos, model in zip(positions, models, strict=True):
        mean_val = np.mean(durations[model])
        ax.annotate(
            f"μ={mean_val:.1f}",
            xy=(pos, max(durations[model]) + 0.5),
            ha="center",
            fontsize=9,
        )

    save_figure(fig, output_path)

    return output_path


def generate_all_evaluation_plots(mock: bool = True) -> list[Path]:
    """
    Generate all evaluation plots.

    Args:
        mock: If True, use mock data

    Returns:
        List of paths to generated figures
    """
    paths = [
        plot_lead_time_bar_chart(),
        plot_false_positive_rate(),
        plot_false_alarm_duration(),
    ]

    return paths
