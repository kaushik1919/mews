"""
Architecture Diagram and Demo Visualization.

MEWS Phase 5B: System architecture block diagram and demo snapshot.

Generates:
    - mews_architecture.png
    - daily_risk_snapshot.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .config import (
    ARCHITECTURE_DIR,
    COLORS,
    DEMO_DIR,
    SIZES,
    create_figure,
    save_figure,
)


def plot_architecture_diagram(
    output_path: Path | None = None,
) -> Path:
    """
    Generate MEWS system architecture block diagram.

    Shows data flow from sources through features to risk engine.

    Args:
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = ARCHITECTURE_DIR / "mews_architecture.png"

    fig, ax = create_figure("architecture")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Color scheme
    source_color = "#E8F4FD"
    ingestion_color = "#D4E8D4"
    feature_color = "#FFF3CD"
    risk_color = "#F8D7DA"
    output_color = "#D1ECF1"

    border_color = COLORS["secondary"]

    def draw_box(x, y, w, h, label, color, fontsize=10):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + w/2, y + h/2, label,
            ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold" if fontsize > 9 else "normal",
            wrap=True,
        )

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": border_color,
                "lw": 1.5,
                "connectionstyle": "arc3,rad=0",
            },
        )

    # Layer labels
    ax.text(0.5, 7.5, "Data Sources", fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.text(3.2, 7.5, "Ingestion", fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.text(5.8, 7.5, "Features", fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.text(9.0, 7.5, "Risk Engine", fontsize=11, fontweight="bold", color=COLORS["primary"])
    ax.text(12.0, 7.5, "Output", fontsize=11, fontweight="bold", color=COLORS["primary"])

    # Data Sources (left column)
    draw_box(0.3, 5.5, 2.0, 0.8, "Market Prices", source_color)
    draw_box(0.3, 4.3, 2.0, 0.8, "Volatility Indices", source_color)
    draw_box(0.3, 3.1, 2.0, 0.8, "Macro Rates", source_color)
    draw_box(0.3, 1.9, 2.0, 0.8, "Financial News", source_color)

    # Ingestion layer
    draw_box(3.0, 3.5, 2.0, 1.5, "Adapters\n& Alignment", ingestion_color)

    # Feature Services
    draw_box(5.7, 5.5, 2.0, 0.8, "Numeric\n(Vol, DD)", feature_color, 9)
    draw_box(5.7, 4.3, 2.0, 0.8, "Graph\n(Corr)", feature_color, 9)
    draw_box(5.7, 3.1, 2.0, 0.8, "Sentiment\n(NLP)", feature_color, 9)
    draw_box(5.7, 1.9, 2.0, 0.8, "Liquidity", feature_color, 9)

    # Risk Engine
    draw_box(8.5, 5.3, 2.4, 1.2, "Heuristic\nEngine", risk_color)
    draw_box(8.5, 3.7, 2.4, 1.2, "ML Engine\n(RF/GB)", risk_color)
    draw_box(8.5, 2.1, 2.4, 1.2, "Ensemble\n& Calibration", risk_color)

    # Output
    draw_box(11.7, 3.7, 2.0, 1.2, "Risk Score\n& Regime", output_color)
    draw_box(11.7, 2.1, 2.0, 1.2, "Explanations\n& Report", output_color)

    # Arrows: Sources -> Ingestion
    for y in [5.9, 4.7, 3.5, 2.3]:
        draw_arrow(2.3, y, 3.0, 4.25)

    # Arrows: Ingestion -> Features
    draw_arrow(5.0, 4.5, 5.7, 5.9)
    draw_arrow(5.0, 4.25, 5.7, 4.7)
    draw_arrow(5.0, 4.0, 5.7, 3.5)
    draw_arrow(5.0, 3.75, 5.7, 2.3)

    # Arrows: Features -> Risk Engine
    draw_arrow(7.7, 5.9, 8.5, 5.9)
    draw_arrow(7.7, 4.7, 8.5, 4.3)
    draw_arrow(7.7, 3.5, 8.5, 4.0)
    draw_arrow(7.7, 2.3, 8.5, 2.7)

    # Arrows: Heuristic/ML -> Ensemble
    draw_arrow(9.7, 5.3, 9.7, 3.3)
    draw_arrow(9.7, 3.7, 9.7, 3.3)

    # Arrows: Ensemble -> Outputs
    draw_arrow(10.9, 2.7, 11.7, 4.0)
    draw_arrow(10.9, 2.7, 11.7, 2.7)

    # Title
    ax.text(7, 0.4, "MEWS: Market Early Warning System",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["primary"])

    save_figure(fig, output_path)

    return output_path


def plot_daily_risk_snapshot(
    output_path: Path | None = None,
) -> Path:
    """
    Generate a daily risk snapshot demo plot.

    Shows a single-day risk report with key indicators.

    Args:
        output_path: Output path (uses default if None)

    Returns:
        Path to saved figure
    """
    if output_path is None:
        output_path = DEMO_DIR / "daily_risk_snapshot.png"

    np.random.seed(400)

    fig = plt.figure(figsize=SIZES["double"])

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Risk gauge (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    risk_score = 0.67
    regime = "High Risk"
    regime_color = COLORS["high_risk"]

    # Simple gauge representation
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    ax1.fill_between(theta, 0, r, alpha=0.1, color=COLORS["secondary"])

    # Risk arc
    risk_theta = np.linspace(0, np.pi * risk_score, 50)
    ax1.fill_between(risk_theta, 0.7, 1.0, alpha=0.8, color=regime_color)

    ax1.set_xlim(-0.2, np.pi + 0.2)
    ax1.set_ylim(-0.3, 1.2)
    ax1.axis("off")
    ax1.text(np.pi/2, 0.3, f"{risk_score:.0%}", ha="center", fontsize=24, fontweight="bold", color=regime_color)
    ax1.text(np.pi/2, -0.1, regime, ha="center", fontsize=12, color=regime_color)
    ax1.set_title("Ensemble Risk Score", fontsize=11, fontweight="bold")

    # Panel 2: Recent trend (top center)
    ax2 = fig.add_subplot(gs[0, 1])

    days = 30
    scores = 0.45 + 0.15 * np.sin(np.linspace(0, 2*np.pi, days)) + np.random.normal(0, 0.03, days)
    scores[-5:] += 0.15  # Recent spike
    scores = np.clip(scores, 0.1, 0.9)

    ax2.plot(range(days), scores, color=COLORS["ensemble"], linewidth=2)
    ax2.fill_between(range(days), scores, alpha=0.3, color=COLORS["ensemble"])
    ax2.axhline(y=0.50, color=COLORS["high_risk"], linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Risk Score")
    ax2.set_xlabel("Days")
    ax2.set_title("30-Day Trend", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1)

    # Panel 3: Component breakdown (top right)
    ax3 = fig.add_subplot(gs[0, 2])

    components = ["Heuristic", "ML (RF)", "ML (GB)"]
    values = [0.58, 0.72, 0.69]
    colors = [COLORS["heuristic"], COLORS["ml"], "#6A5ACD"]

    bars = ax3.barh(components, values, color=colors, alpha=0.8)
    ax3.axvline(x=0.50, color=COLORS["secondary"], linestyle="--", linewidth=1)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Score")
    ax3.set_title("Component Scores", fontsize=11, fontweight="bold")

    for bar, val in zip(bars, values, strict=True):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.0%}", va="center", fontsize=10)

    # Panel 4: Top drivers (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])

    drivers = ["Realized Vol (20d)", "Avg Correlation", "Max Drawdown", "VIX Level"]
    contributions = [0.22, 0.18, 0.12, 0.08]
    driver_colors = [COLORS["volatility"], COLORS["correlation"], COLORS["drawdown"], COLORS["volatility"]]

    ax4.barh(drivers, contributions, color=driver_colors, alpha=0.8)
    ax4.set_xlabel("SHAP Contribution")
    ax4.set_title("Top Risk Drivers", fontsize=11, fontweight="bold")

    # Panel 5: Feature values (bottom center)
    ax5 = fig.add_subplot(gs[1, 1])

    features = ["Vol 20d", "Corr", "DD 60d", "Sent"]
    current = [0.28, 0.72, -0.15, -0.12]
    historical = [0.18, 0.55, -0.08, 0.05]

    x = np.arange(len(features))
    width = 0.35

    ax5.bar(x - width/2, current, width, label="Current", color=COLORS["high_risk"], alpha=0.8)
    ax5.bar(x + width/2, historical, width, label="Avg (1Y)", color=COLORS["secondary"], alpha=0.6)
    ax5.set_xticks(x)
    ax5.set_xticklabels(features)
    ax5.set_ylabel("Value")
    ax5.set_title("Feature Comparison", fontsize=11, fontweight="bold")
    ax5.legend(loc="upper right", fontsize=9)

    # Panel 6: Regime history (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])

    regime_counts = {"Low": 45, "Moderate": 30, "High": 18, "Extreme": 7}
    colors = [COLORS["low_risk"], COLORS["moderate_risk"], COLORS["high_risk"], COLORS["extreme_risk"]]

    ax6.pie(
        regime_counts.values(),
        labels=regime_counts.keys(),
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    ax6.set_title("Regime Distribution (1Y)", fontsize=11, fontweight="bold")

    # Main title
    fig.suptitle("MEWS Daily Risk Report — 2023-10-15", fontsize=14, fontweight="bold", y=0.98)

    save_figure(fig, output_path)

    return output_path


def generate_all_architecture_plots(mock: bool = True) -> list[Path]:
    """
    Generate architecture and demo plots.

    Args:
        mock: If True, use mock data (always true for these)

    Returns:
        List of paths to generated figures
    """
    paths = [
        plot_architecture_diagram(),
        plot_daily_risk_snapshot(),
    ]

    return paths
