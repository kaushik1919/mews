"""
MEWS Visualization Module.

Phase 5B: Documentation figures and plots.

Usage:
    python -m visualization.run_all --mock
"""

from .architecture import (
    generate_all_architecture_plots,
    plot_architecture_diagram,
    plot_daily_risk_snapshot,
)
from .captions import (
    ALL_CAPTIONS,
    format_caption_for_markdown,
    get_caption,
)
from .config import (
    ARCHITECTURE_DIR,
    COLORS,
    DEMO_DIR,
    EVALUATION_DIR,
    FEATURES_DIR,
    FIGURES_DIR,
    REGIME_COLORS,
    RISK_ENGINE_DIR,
    SIZES,
    add_regime_background,
    apply_style,
    create_figure,
    create_subplots,
    ensure_dirs,
    save_figure,
    shade_crisis_periods,
)
from .evaluation import (
    generate_all_evaluation_plots,
    plot_false_alarm_duration,
    plot_false_positive_rate,
    plot_lead_time_bar_chart,
)
from .features import (
    generate_all_feature_plots,
    plot_correlation_timeseries,
    plot_sentiment_vs_market,
    plot_volatility_and_drawdown,
)
from .risk import (
    generate_all_risk_plots,
    plot_calibration_curve,
    plot_ensemble_vs_components,
    plot_heuristic_risk_score,
    plot_ml_vs_heuristic,
    plot_shap_importance,
)

__all__ = [
    # Directories
    "FIGURES_DIR",
    "ARCHITECTURE_DIR",
    "FEATURES_DIR",
    "RISK_ENGINE_DIR",
    "EVALUATION_DIR",
    "DEMO_DIR",
    # Config
    "COLORS",
    "REGIME_COLORS",
    "SIZES",
    "ensure_dirs",
    "apply_style",
    "create_figure",
    "create_subplots",
    "shade_crisis_periods",
    "add_regime_background",
    "save_figure",
    # Architecture
    "plot_architecture_diagram",
    "plot_daily_risk_snapshot",
    "generate_all_architecture_plots",
    # Features
    "plot_volatility_and_drawdown",
    "plot_sentiment_vs_market",
    "plot_correlation_timeseries",
    "generate_all_feature_plots",
    # Risk
    "plot_heuristic_risk_score",
    "plot_ml_vs_heuristic",
    "plot_calibration_curve",
    "plot_ensemble_vs_components",
    "plot_shap_importance",
    "generate_all_risk_plots",
    # Evaluation
    "plot_lead_time_bar_chart",
    "plot_false_positive_rate",
    "plot_false_alarm_duration",
    "generate_all_evaluation_plots",
    # Captions
    "ALL_CAPTIONS",
    "get_caption",
    "format_caption_for_markdown",
]
