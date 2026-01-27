"""
MEWS Visualization Module.

Phase 5B: Documentation figures and plots.

Usage:
    python -m visualization.run_all --mock
"""

from .config import (
    FIGURES_DIR,
    ARCHITECTURE_DIR,
    FEATURES_DIR,
    RISK_ENGINE_DIR,
    EVALUATION_DIR,
    DEMO_DIR,
    COLORS,
    REGIME_COLORS,
    SIZES,
    ensure_dirs,
    apply_style,
    create_figure,
    create_subplots,
    shade_crisis_periods,
    add_regime_background,
    save_figure,
)

from .architecture import (
    plot_architecture_diagram,
    plot_daily_risk_snapshot,
    generate_all_architecture_plots,
)

from .features import (
    plot_volatility_and_drawdown,
    plot_sentiment_vs_market,
    plot_correlation_timeseries,
    generate_all_feature_plots,
)

from .risk import (
    plot_heuristic_risk_score,
    plot_ml_vs_heuristic,
    plot_calibration_curve,
    plot_ensemble_vs_components,
    plot_shap_importance,
    generate_all_risk_plots,
)

from .evaluation import (
    plot_lead_time_bar_chart,
    plot_false_positive_rate,
    plot_false_alarm_duration,
    generate_all_evaluation_plots,
)

from .captions import (
    ALL_CAPTIONS,
    get_caption,
    format_caption_for_markdown,
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
