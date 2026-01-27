"""
Visualization Configuration.

MEWS Phase 5B: Shared plotting style, colors, and fonts.

All plots use Matplotlib with a consistent, publication-quality style.
Deterministic output. No interactive backends.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for deterministic output
matplotlib.use("Agg")


# ==============================================================================
# PATHS
# ==============================================================================

# Base output directory for figures
FIGURES_DIR = Path("figures")

# Subdirectories
ARCHITECTURE_DIR = FIGURES_DIR / "architecture"
FEATURES_DIR = FIGURES_DIR / "features"
RISK_ENGINE_DIR = FIGURES_DIR / "risk_engine"
EVALUATION_DIR = FIGURES_DIR / "evaluation"
DEMO_DIR = FIGURES_DIR / "demo"

ALL_DIRS = [
    FIGURES_DIR,
    ARCHITECTURE_DIR,
    FEATURES_DIR,
    RISK_ENGINE_DIR,
    EVALUATION_DIR,
    DEMO_DIR,
]


def ensure_dirs() -> None:
    """Create all figure directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# COLORS
# ==============================================================================

# Neutral, professional color palette
COLORS = {
    # Primary colors
    "primary": "#2C3E50",      # Dark blue-gray
    "secondary": "#7F8C8D",    # Medium gray
    "accent": "#E74C3C",       # Red for alerts/crises

    # Risk regime colors
    "low_risk": "#27AE60",     # Green
    "moderate_risk": "#F39C12", # Yellow/Orange
    "high_risk": "#E74C3C",    # Red
    "extreme_risk": "#8E44AD", # Purple

    # Model colors
    "heuristic": "#3498DB",    # Blue
    "ml": "#2ECC71",           # Green
    "ensemble": "#9B59B6",     # Purple
    "random_forest": "#2ECC71",
    "xgboost": "#1ABC9C",

    # Feature colors
    "volatility": "#E74C3C",
    "correlation": "#3498DB",
    "sentiment": "#F39C12",
    "liquidity": "#1ABC9C",
    "drawdown": "#9B59B6",

    # Crisis shading
    "crisis_fill": "#FADBD8",  # Light red
    "crisis_edge": "#E74C3C",

    # Grid and background
    "grid": "#ECF0F1",
    "background": "#FFFFFF",
}

REGIME_COLORS = {
    "LOW_RISK": COLORS["low_risk"],
    "MODERATE_RISK": COLORS["moderate_risk"],
    "HIGH_RISK": COLORS["high_risk"],
    "EXTREME_RISK": COLORS["extreme_risk"],
}


# ==============================================================================
# FIGURE SIZES
# ==============================================================================

# Standard figure sizes (width, height) in inches
SIZES = {
    "single": (8, 5),          # Single panel
    "wide": (12, 5),           # Wide single panel
    "double": (12, 8),         # Two-panel vertical
    "square": (8, 8),          # Square
    "architecture": (14, 8),   # Architecture diagram
    "heatmap": (10, 8),        # Correlation heatmap
}


# ==============================================================================
# STYLE CONFIGURATION
# ==============================================================================

STYLE = {
    # Font settings
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,

    # Axes
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 1.0,
    "axes.edgecolor": COLORS["primary"],
    "axes.labelcolor": COLORS["primary"],
    "axes.facecolor": COLORS["background"],

    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": COLORS["primary"],
    "ytick.color": COLORS["primary"],

    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": COLORS["secondary"],
    "grid.linestyle": "--",

    # Legend
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,

    # Figure
    "figure.facecolor": COLORS["background"],
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}


def apply_style() -> None:
    """Apply the MEWS plotting style globally."""
    plt.rcParams.update(STYLE)


def create_figure(size_key: str = "single") -> tuple[plt.Figure, plt.Axes]:
    """
    Create a new figure with MEWS style.

    Args:
        size_key: Key from SIZES dict

    Returns:
        Tuple of (figure, axes)
    """
    apply_style()
    fig, ax = plt.subplots(figsize=SIZES.get(size_key, SIZES["single"]))
    return fig, ax


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    size_key: str = "double",
) -> tuple[plt.Figure, list]:
    """
    Create subplots with MEWS style.

    Args:
        nrows: Number of rows
        ncols: Number of columns
        size_key: Key from SIZES dict

    Returns:
        Tuple of (figure, axes_list)
    """
    apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=SIZES.get(size_key, SIZES["double"]))
    return fig, axes


def shade_crisis_periods(
    ax: plt.Axes,
    crisis_periods: list[tuple[str, str]],
    alpha: float = 0.2,
) -> None:
    """
    Add shaded regions for crisis periods.

    Args:
        ax: Matplotlib axes
        crisis_periods: List of (start_date, end_date) tuples as strings
        alpha: Opacity of shading
    """
    import pandas as pd

    for start, end in crisis_periods:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            alpha=alpha,
            facecolor=COLORS["crisis_fill"],
            edgecolor=COLORS["crisis_edge"],
            linewidth=0.5,
        )


def add_regime_background(
    ax: plt.Axes,
    show_labels: bool = True,
) -> None:
    """
    Add horizontal bands showing regime thresholds.

    Args:
        ax: Matplotlib axes
        show_labels: Whether to add regime labels
    """
    thresholds = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    regimes = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "EXTREME_RISK"]

    for (low, high), regime in zip(thresholds, regimes, strict=True):
        ax.axhspan(low, high, alpha=0.1, color=REGIME_COLORS[regime])
        if show_labels:
            ax.text(
                ax.get_xlim()[1], (low + high) / 2,
                regime.replace("_", " "),
                va="center", ha="left",
                fontsize=8, alpha=0.7,
                color=REGIME_COLORS[regime],
            )


def save_figure(fig: plt.Figure, path: Path, close: bool = True) -> None:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        path: Output path
        close: Whether to close figure after saving
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
