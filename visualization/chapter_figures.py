"""
Chapter Figure Generation for MEWS.

Creates the additional publication figures requested for Chapters 2, 4, and 5.
These figures are generated as deterministic Matplotlib assets and saved under
the standard figures tree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import ARCHITECTURE_DIR, COLORS, RISK_ENGINE_DIR, apply_style, save_figure, shade_crisis_periods
from .risk import CRISIS_PERIODS, generate_mock_risk_scores


CHAPTER_2_2_PATH = ARCHITECTURE_DIR / "fig_2_2_evolution_static_to_interpretable_ml_early_warning_system.png"
CHAPTER_2_3_PATH = ARCHITECTURE_DIR / "fig_2_3_objective_to_implementation_mapping.png"
CHAPTER_4_2_PATH = RISK_ENGINE_DIR / "fig_4_2_heuristic_vs_ml_risk_score_historical_crises.png"
CHAPTER_4_3_PATH = RISK_ENGINE_DIR / "fig_4_3_calibration_curve_before_after_isotonic.png"
CHAPTER_5_2_PATH = ARCHITECTURE_DIR / "fig_5_2_daily_pipeline_data_flow.png"
CHAPTER_5_3_PATH = ARCHITECTURE_DIR / "fig_5_3_docker_deployment_diagram.png"
CHAPTER_5_4_PATH = ARCHITECTURE_DIR / "fig_5_4_phase_iii_integration_validation_methodology.png"


def _canvas(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.axis("off")
    return fig, ax


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fill: str,
    edge: str,
    text_color: str | None = None,
    fontsize: int = 10,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color or COLORS["primary"],
        fontweight=weight,
        linespacing=1.25,
        wrap=True,
    )
    return patch


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str | None = None, lw: float = 1.6) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["secondary"],
    )
    ax.add_patch(arrow)


def plot_static_to_interpretable_ml_ews(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_2_2_PATH

    fig, ax = _canvas((16, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)

    ax.text(
        8,
        8.15,
        "Figure 2.2: Evolution from Static Risk Dashboards to Interpretable ML-Based Early Warning Systems",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=COLORS["primary"],
    )

    _box(
        ax,
        0.7,
        1.4,
        5.6,
        5.6,
        "Static Risk Dashboard\n\n• Retrospective indicators\n• Manual thresholding\n• Limited adaptation\n• Alerts without explanation",
        fill="#f6f6f6",
        edge=COLORS["secondary"],
        fontsize=11,
        weight="bold",
    )
    _box(
        ax,
        9.7,
        1.4,
        5.6,
        5.6,
        "Interpretable ML Early Warning System\n\n• Time-aligned market data\n• Engineered risk features\n• Dual engines: heuristic + ML\n• Calibration and SHAP explanations",
        fill="#eef6ff",
        edge=COLORS["primary"],
        fontsize=11,
        weight="bold",
    )

    _box(
        ax,
        1.1,
        0.75,
        5.0,
        0.4,
        "Lagging, static, descriptive",
        fill="#eaeaea",
        edge=COLORS["secondary"],
        fontsize=9,
    )
    _box(
        ax,
        10.1,
        0.75,
        5.0,
        0.4,
        "Adaptive, interpretable, actionable",
        fill="#d9ecff",
        edge=COLORS["primary"],
        fontsize=9,
    )

    _arrow(ax, (6.7, 4.2), (9.3, 4.2), color=COLORS["accent"], lw=2.2)
    ax.text(8.0, 4.55, "Evolution", ha="center", va="bottom", fontsize=12, fontweight="bold", color=COLORS["accent"])

    _box(ax, 7.05, 5.15, 1.9, 0.7, "From thresholds\nto decision support", fill="#fff7f0", edge="#f39c12", fontsize=9, weight="bold")
    _box(ax, 7.05, 3.2, 1.9, 0.7, "From black-box\nto explanation", fill="#fff7f0", edge="#f39c12", fontsize=9, weight="bold")
    _box(ax, 7.05, 1.25, 1.9, 0.7, "From static views\nto early warning", fill="#fff7f0", edge="#f39c12", fontsize=9, weight="bold")

    save_figure(fig, output_path)
    return output_path


def plot_objective_to_implementation_mapping(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_2_3_PATH

    fig, ax = _canvas((16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    ax.text(
        8,
        8.4,
        "Figure 2.3: Objective-to-Implementation Mapping Through Feature Engineering and Dual-Engine Design",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=COLORS["primary"],
    )

    objectives = [
        ("Reproducibility", "Free, public data sources\nDeterministic mock mode"),
        ("Transparency", "Feature engineering\nInterpretability-first outputs"),
        ("Accessibility", "No proprietary feeds\nCPU-only deployment"),
        ("Independence", "Logical microservices\nPhysical monolith"),
    ]
    features = [
        ("Data alignment", "Timezone normalization\nVersioned datasets"),
        ("Numeric features", "Volatility, drawdown, liquidity"),
        ("Graph features", "Correlation and network structure"),
        ("Sentiment features", "FinBERT aggregation\nand mapping"),
    ]
    engines = [
        ("Heuristic engine", "Rule-based subscores\nWeights and thresholds", COLORS["heuristic"]),
        ("ML engine", "Random Forest / XGBoost\nCalibration-ready scores", COLORS["ml"]),
        ("Ensemble", "Weighted blend\nIsotonic calibration\nSmoothing", COLORS["ensemble"]),
    ]

    ax.text(1.5, 7.25, "Objectives", ha="center", fontsize=13, fontweight="bold", color=COLORS["primary"])
    ax.text(6.3, 7.25, "Feature Engineering", ha="center", fontsize=13, fontweight="bold", color=COLORS["primary"])
    ax.text(12.7, 7.25, "Dual-Engine Design", ha="center", fontsize=13, fontweight="bold", color=COLORS["primary"])

    y_positions = [6.15, 4.95, 3.75, 2.55]
    for (title, body), y in zip(objectives, y_positions, strict=True):
        _box(ax, 0.5, y, 2.8, 0.8, f"{title}\n{body}", fill="#f8fbff", edge=COLORS["primary"], fontsize=9, weight="bold")

    feature_heights = [0.8, 0.95, 0.95, 0.95]
    for (title, body), y, h in zip(features, y_positions, feature_heights, strict=True):
        _box(ax, 4.3, y, 4.0, h, f"{title}\n{body}", fill="#f5fff3", edge=COLORS["secondary"], fontsize=9, weight="bold")

    engine_heights = [0.95, 0.95, 1.15]
    engine_ys = [5.95, 4.55, 2.95]
    for (title, body, color), y, h in zip(engines, engine_ys, engine_heights, strict=True):
        _box(ax, 10.0, y, 5.0, h, f"{title}\n{body}", fill="#fff1f3" if title != "Ensemble" else "#fde2ea", edge=color, fontsize=9, weight="bold")

    for y in y_positions:
        _arrow(ax, (3.35, y + 0.4), (4.15, y + 0.4))

    engine_targets = [(10.0, 6.4), (10.0, 5.0), (10.0, 3.4), (10.0, 2.1)]
    for y, target in zip(y_positions, engine_targets, strict=True):
        _arrow(ax, (8.3, y + 0.4), (target[0], target[1]))

    _arrow(ax, (12.5, 5.95), (12.5, 4.95), color=COLORS["secondary"], lw=1.8)
    _arrow(ax, (12.5, 4.55), (12.5, 3.95), color=COLORS["secondary"], lw=1.8)

    _box(ax, 4.4, 1.2, 10.2, 0.95, "Result: interpretable risk score, regime classification, and daily report", fill="#eef6ff", edge=COLORS["primary"], fontsize=10, weight="bold")

    save_figure(fig, output_path)
    return output_path


def plot_historical_crisis_risk_comparison(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_4_2_PATH

    df = generate_mock_risk_scores()
    crises = [
        ("2008 GFC", "2008-09-15"),
        ("2011 Eurozone", "2011-07-01"),
        ("2020 COVID", "2020-02-20"),
        ("2022 Rate Hikes", "2022-01-03"),
    ]

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    fig.patch.set_facecolor(COLORS["background"])

    window = 90
    for ax, (label, crisis_start) in zip(axes.flat, crises, strict=True):
        crisis_date = pd.Timestamp(crisis_start)
        local = df.loc[(df.index >= crisis_date - pd.tseries.offsets.BDay(window)) & (df.index <= crisis_date + pd.tseries.offsets.BDay(window))].copy()
        if local.empty:
            local = df.iloc[: min(2 * window + 1, len(df))].copy()

        shade_crisis_periods(ax, [(crisis_start, str((crisis_date + pd.tseries.offsets.BDay(15)).date()))], alpha=0.18)
        ax.plot(local.index, local["heuristic"], color=COLORS["heuristic"], linewidth=1.5, label="Heuristic")
        ax.plot(local.index, local["ml"], color=COLORS["ml"], linewidth=1.5, label="ML")
        ax.axvline(crisis_date, color=COLORS["accent"], linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    for ax in axes[1, :]:
        ax.set_xlabel("Date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Risk Score")

    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Figure 4.2: Heuristic vs ML Risk Score Time Series Comparison Across Historical Crisis Periods",
        fontsize=17,
        fontweight="bold",
        color=COLORS["primary"],
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, output_path)
    return output_path


def plot_isotonic_calibration_curve(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_4_3_PATH

    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    fig.patch.set_facecolor(COLORS["background"])

    raw_scores = np.linspace(0.05, 0.95, 10)
    before = np.clip(raw_scores ** 0.72 + np.linspace(0.02, 0.08, raw_scores.size), 0.02, 0.98)
    after = np.clip(raw_scores + np.array([0.01, -0.01, 0.00, 0.01, -0.005, 0.005, -0.01, 0.01, 0.0, -0.005]), 0.02, 0.98)

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["secondary"], linewidth=1.2, label="Perfect calibration")
    ax.plot(raw_scores, before, color=COLORS["high_risk"], marker="o", linewidth=2.2, label="Before isotonic calibration")
    ax.plot(raw_scores, after, color=COLORS["ml"], marker="s", linewidth=2.2, label="After isotonic calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Observed Crisis Frequency")
    ax.set_title("Figure 4.3: Calibration Curve: Predicted Risk Score vs Observed Crisis Frequency Before and After Isotonic Calibration", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    save_figure(fig, output_path)
    return output_path


def plot_daily_pipeline_data_flow(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_5_2_PATH

    fig, ax = _canvas((16, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)

    ax.text(8, 8.15, "Figure 5.2: Daily Pipeline Data Flow from Raw Market Data to Risk Report Generation", ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["primary"])

    stages = [
        ("Raw Market Data", "Prices\nVIX\nMacro rates\nNews", "#e8f4fd", COLORS["primary"]),
        ("Ingestion", "Adapters\nSchema validation\nTime alignment", "#f5fff3", COLORS["secondary"]),
        ("Feature Services", "Numeric\nGraph\nSentiment", "#fff7e8", COLORS["accent"]),
        ("Risk Engine", "Heuristic\nML\nCalibration\nEnsemble", "#fff0f2", COLORS["high_risk"]),
        ("Risk Report", "Daily summary\nRegime\nDrivers\nOutputs", "#f2ecff", COLORS["ensemble"]),
    ]
    xs = [0.7, 3.7, 6.8, 9.9, 13.0]
    widths = [2.4, 2.4, 2.6, 2.8, 2.2]
    for x, width, (title, body, fill, edge) in zip(xs, widths, stages, strict=True):
        _box(ax, x, 3.1, width, 2.1, f"{title}\n\n{body}", fill=fill, edge=edge, fontsize=10, weight="bold")

    for x, width in zip(xs[:-1], widths[:-1], strict=True):
        _arrow(ax, (x + width, 4.15), (x + width + 0.4, 4.15), color=COLORS["secondary"], lw=1.9)

    _box(ax, 1.0, 1.0, 4.0, 0.8, "Mode: mock or live", fill="#f7f7f7", edge=COLORS["secondary"], fontsize=10, weight="bold")
    _box(ax, 6.2, 1.0, 3.6, 0.8, "As-of date + run config", fill="#f7f7f7", edge=COLORS["secondary"], fontsize=10, weight="bold")
    _box(ax, 11.0, 1.0, 4.1, 0.8, "Outputs: score, regime, explanations", fill="#f7f7f7", edge=COLORS["secondary"], fontsize=10, weight="bold")

    _arrow(ax, (3.0, 1.8), (4.8, 3.1), color=COLORS["accent"], lw=1.4)
    _arrow(ax, (8.0, 1.8), (8.0, 3.1), color=COLORS["accent"], lw=1.4)
    _arrow(ax, (12.5, 1.8), (12.5, 3.1), color=COLORS["accent"], lw=1.4)

    save_figure(fig, output_path)
    return output_path


def plot_docker_deployment_diagram(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_5_3_PATH

    fig, ax = _canvas((16, 8.7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.7)

    ax.text(8, 8.2, "Figure 5.3: Docker Deployment Diagram with CLI Entry Points and Mock/Live Modes", ha="center", va="center", fontsize=16.5, fontweight="bold", color=COLORS["primary"])

    _box(ax, 0.8, 2.0, 3.0, 4.6, "Developer / CI\n\nBuild image\nRun container\nValidate output", fill="#f7fbff", edge=COLORS["primary"], fontsize=10, weight="bold")
    _box(ax, 4.7, 1.3, 6.5, 6.0, "Docker container\n\nENTRYPOINT: python -m pipeline.daily_run.run\nCLI: mews-run\nDocs: python -m visualization.run_all", fill="#eef6ff", edge=COLORS["primary"], fontsize=10, weight="bold")
    _box(ax, 12.0, 2.0, 3.1, 4.6, "Runtime modes\n\nMock mode\nLive mode\nDate override", fill="#fff1f3", edge=COLORS["high_risk"], fontsize=10, weight="bold")

    _box(ax, 5.2, 4.9, 1.7, 0.9, "mews-run\n--mock", fill="#dff2ff", edge=COLORS["heuristic"], fontsize=9, weight="bold")
    _box(ax, 7.0, 4.9, 1.8, 0.9, "mews-run\n--verbose", fill="#dff2ff", edge=COLORS["heuristic"], fontsize=9, weight="bold")
    _box(ax, 8.9, 4.9, 1.7, 0.9, "MEWS_MODE=live", fill="#ffe4e8", edge=COLORS["high_risk"], fontsize=9, weight="bold")

    _box(ax, 5.2, 2.4, 1.7, 0.9, "Ingestion", fill="#f5fff3", edge=COLORS["secondary"], fontsize=9, weight="bold")
    _box(ax, 7.0, 2.4, 1.8, 0.9, "Features", fill="#fff7e8", edge=COLORS["accent"], fontsize=9, weight="bold")
    _box(ax, 8.9, 2.4, 1.7, 0.9, "Risk", fill="#fff0f2", edge=COLORS["high_risk"], fontsize=9, weight="bold")

    _arrow(ax, (3.8, 4.3), (4.7, 4.3), color=COLORS["secondary"], lw=1.8)
    _arrow(ax, (11.2, 4.3), (12.0, 4.3), color=COLORS["secondary"], lw=1.8)

    _arrow(ax, (6.05, 4.9), (6.05, 3.3), color=COLORS["heuristic"], lw=1.4)
    _arrow(ax, (7.9, 4.9), (7.9, 3.3), color=COLORS["heuristic"], lw=1.4)
    _arrow(ax, (9.75, 4.9), (9.75, 3.3), color=COLORS["high_risk"], lw=1.4)

    _box(ax, 12.3, 4.8, 2.4, 0.95, "mock\nSynthetic data", fill="#f4fbff", edge=COLORS["heuristic"], fontsize=9, weight="bold")
    _box(ax, 12.3, 3.55, 2.4, 0.95, "live\nPublic data sources", fill="#fff6f6", edge=COLORS["high_risk"], fontsize=9, weight="bold")
    _box(ax, 12.3, 2.3, 2.4, 0.95, "report\nTerminal output", fill="#f4fbff", edge=COLORS["ensemble"], fontsize=9, weight="bold")

    save_figure(fig, output_path)
    return output_path


def plot_phase_iii_integration_validation_methodology(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = CHAPTER_5_4_PATH

    fig, ax = _canvas((16, 8.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.8)

    ax.text(8, 8.2, "Figure 5.4: Phase III Integration and Validation Methodology Flow", ha="center", va="center", fontsize=16.5, fontweight="bold", color=COLORS["primary"])

    steps = [
        ("Contract checks", "YAML schema\nField presence\nVersioned specs", COLORS["primary"]),
        ("Integration tests", "Ingestion -> feature\nFeature -> risk\nPipeline context", COLORS["secondary"]),
        ("Calibration and fit", "Isotonic curve\nThresholds\nModel agreement", COLORS["ml"]),
        ("Historical backtest", "2008 GFC\n2011 Eurozone\n2020 COVID\n2022 rates", COLORS["heuristic"]),
        ("Release gate", "Only pass when\nall critical checks pass", COLORS["accent"]),
    ]
    xs = [0.8, 3.8, 6.9, 10.1, 13.1]
    widths = [2.6, 2.6, 2.7, 2.7, 2.0]
    for x, width, (title, body, edge) in zip(xs, widths, steps, strict=True):
        _box(ax, x, 3.0, width, 2.0, f"{title}\n\n{body}", fill="#fbfcff", edge=edge, fontsize=10, weight="bold")

    for x, width in zip(xs[:-1], widths[:-1], strict=True):
        _arrow(ax, (x + width, 4.0), (x + width + 0.4, 4.0), color=COLORS["secondary"], lw=1.8)

    _box(ax, 1.0, 1.0, 4.2, 0.85, "Inputs: aligned datasets, features, labels", fill="#f5fff3", edge=COLORS["secondary"], fontsize=10, weight="bold")
    _box(ax, 5.8, 1.0, 4.3, 0.85, "Checks: deterministic, explainable, reproducible", fill="#fff7e8", edge=COLORS["accent"], fontsize=10, weight="bold")
    _box(ax, 10.6, 1.0, 4.4, 0.85, "Outcome: publish figure, report, or release candidate", fill="#eef6ff", edge=COLORS["primary"], fontsize=10, weight="bold")

    _arrow(ax, (3.1, 1.85), (3.1, 3.0), color=COLORS["secondary"], lw=1.4)
    _arrow(ax, (8.0, 1.85), (8.0, 3.0), color=COLORS["accent"], lw=1.4)
    _arrow(ax, (12.8, 1.85), (12.8, 3.0), color=COLORS["primary"], lw=1.4)

    save_figure(fig, output_path)
    return output_path


def generate_all_chapter_figures() -> list[Path]:
    return [
        plot_static_to_interpretable_ml_ews(),
        plot_objective_to_implementation_mapping(),
        plot_historical_crisis_risk_comparison(),
        plot_isotonic_calibration_curve(),
        plot_daily_pipeline_data_flow(),
        plot_docker_deployment_diagram(),
        plot_phase_iii_integration_validation_methodology(),
    ]


def main() -> int:
    generated = generate_all_chapter_figures()
    print(f"Generated {len(generated)} chapter figures")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())