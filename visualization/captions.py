"""
Publication-Quality Captions.

MEWS Phase 5B: Figure captions for documentation and README.
"""

from __future__ import annotations

# Architecture captions
ARCHITECTURE_CAPTIONS = {
    "mews_architecture.png": (
        "**Figure 1: MEWS System Architecture.** "
        "Data flows from four sources (market prices, volatility indices, macro rates, financial news) "
        "through ingestion adapters with time alignment, into three feature services (numeric, graph, sentiment). "
        "Features feed into dual risk engines (rule-based heuristic and ML-based), which are combined via "
        "calibrated ensemble to produce final risk scores, regime classifications, and explanatory reports."
    ),
}

# Feature captions
FEATURE_CAPTIONS = {
    "volatility_and_drawdown_timeseries.png": (
        "**Figure 2: Realized Volatility and Maximum Drawdown.** "
        "Top panel shows 20-day rolling realized volatility across major market indices. "
        "Bottom panel shows 60-day maximum drawdown. Gray shaded regions indicate known crisis periods "
        "(2008 GFC, 2011 Eurozone, 2020 COVID, 2022 Rate Hikes). Both metrics exhibit clear spikes "
        "during market stress, validating their role as leading risk indicators."
    ),
    "news_sentiment_vs_market.png": (
        "**Figure 3: News Sentiment vs Market Returns.** "
        "Dual-axis plot comparing 5-day aggregated news sentiment (left axis) with cumulative market returns (right axis). "
        "Sentiment deterioration often precedes or coincides with market drawdowns, "
        "demonstrating the predictive value of NLP-based sentiment signals."
    ),
    "avg_correlation_timeseries.png": (
        "**Figure 4: Average Pairwise Correlation.** "
        "Rolling 20-day average pairwise correlation across major equity indices. "
        "Elevated correlation (above 0.7 threshold) indicates regime of systemic risk where "
        "diversification benefits diminish. Crisis periods show characteristic correlation spikes."
    ),
}

# Risk engine captions
RISK_ENGINE_CAPTIONS = {
    "heuristic_risk_score.png": (
        "**Figure 5: Heuristic Risk Score.** "
        "Rule-based risk score combining normalized volatility, drawdown, correlation, and sentiment subscores "
        "with interpretable weights. Horizontal dashed lines indicate regime thresholds: "
        "0.25 (low/moderate), 0.50 (moderate/high), 0.75 (high/extreme). "
        "Score reliably elevates during crisis periods with minimal false alarms."
    ),
    "ml_vs_heuristic_comparison.png": (
        "**Figure 6: ML vs Heuristic Comparison.** "
        "Top panel: Time series comparison of heuristic (rule-based) and ML (Random Forest) risk scores. "
        "Bottom panel: Scatter plot showing score agreement. ML scores tend to be more reactive, "
        "capturing regime transitions earlier, while heuristic scores provide stable baselines."
    ),
    "calibration_curve.png": (
        "**Figure 7: Calibration Curve.** "
        "Reliability diagram showing predicted risk scores vs observed crisis frequency. "
        "Before calibration (red), ML models exhibit overconfidence. "
        "After isotonic calibration (blue), scores align with the diagonal, "
        "ensuring probabilistic interpretability of risk levels."
    ),
    "ensemble_vs_components.png": (
        "**Figure 8: Ensemble vs Component Models.** "
        "Calibrated ensemble (bold line) combining heuristic (35% weight) and ML (65% weight) components. "
        "Ensemble achieves best of both: heuristic stability and ML reactivity. "
        "Exponential smoothing reduces day-to-day noise while preserving signal during regime transitions."
    ),
    "shap_global_importance.png": (
        "**Figure 9: Global Feature Importance (SHAP).** "
        "Mean absolute SHAP values across test set, showing each feature's average contribution to predictions. "
        "Volatility and correlation features dominate, confirming that market microstructure signals "
        "drive risk predictions. Sentiment features provide complementary information during news-driven events."
    ),
}

# Evaluation captions
EVALUATION_CAPTIONS = {
    "lead_time_bar_chart.png": (
        "**Figure 10: Lead Time Analysis.** "
        "Business days of advance warning before each crisis peak, using 0.60 alert threshold. "
        "Ensemble model consistently provides longest lead times across all four historical crises, "
        "ranging from 11 days (COVID) to 25 days (GFC). ML outperforms pure heuristic in all cases."
    ),
    "false_positive_rate_by_threshold.png": (
        "**Figure 11: False Positive Rate by Threshold.** "
        "Trade-off between alert sensitivity and false alarm rate. "
        "Ensemble achieves lowest false positive rate at each threshold level. "
        "Vertical dashed line indicates default threshold (0.60), balancing ~7% FPR with adequate lead time."
    ),
    "false_alarm_duration.png": (
        "**Figure 12: False Alarm Duration.** "
        "Distribution of consecutive days in false alert state. "
        "Ensemble produces shortest median streak duration (2-3 days), "
        "reducing alert fatigue compared to heuristic or standalone ML models."
    ),
}

# Demo captions
DEMO_CAPTIONS = {
    "daily_risk_snapshot.png": (
        "**Figure 13: Daily Risk Report.** "
        "Example output from MEWS daily pipeline showing: (1) current ensemble risk score and regime, "
        "(2) 30-day trend, (3) component model breakdown, (4) top risk drivers via SHAP, "
        "(5) current vs historical feature values, and (6) regime distribution over trailing year. "
        "This dashboard enables rapid risk assessment for portfolio managers."
    ),
}

# Chapter figures
CHAPTER_CAPTIONS = {
    "fig_2_2_evolution_static_to_interpretable_ml_early_warning_system.png": (
        "**Figure 2.2: Evolution from Static Risk Dashboards to Interpretable ML-Based Early Warning Systems.** "
        "A static risk dashboard depends on retrospective indicators and manual thresholds, while the MEWS design "
        "adds time-aligned ingestion, engineered features, dual engines, calibration, and explanations to produce "
        "an adaptive early warning system."
    ),
    "fig_2_3_objective_to_implementation_mapping.png": (
        "**Figure 2.3: Objective-to-Implementation Mapping Through Feature Engineering and Dual-Engine Design.** "
        "Project objectives such as reproducibility, transparency, accessibility, and independence map directly to "
        "free public data sources, deterministic mock mode, feature engineering, and a dual-engine scoring design."
    ),
    "fig_4_2_heuristic_vs_ml_risk_score_historical_crises.png": (
        "**Figure 4.2: Heuristic vs ML Risk Score Time Series Comparison Across Historical Crisis Periods.** "
        "The heuristic and ML scores are compared in crisis-specific windows around the GFC, Eurozone stress, COVID-19, "
        "and the 2022 rate-hike regime. The ML curve is more reactive, while the heuristic score remains smoother."
    ),
    "fig_4_3_calibration_curve_before_after_isotonic.png": (
        "**Figure 4.3: Calibration Curve: Predicted Risk Score vs Observed Crisis Frequency Before and After Isotonic Calibration.** "
        "The before-calibration curve is overconfident relative to the diagonal. Isotonic calibration brings predicted "
        "risk scores into closer alignment with observed crisis frequency."
    ),
    "fig_5_2_daily_pipeline_data_flow.png": (
        "**Figure 5.2: Daily Pipeline Data Flow from Raw Market Data to Risk Report Generation.** "
        "The operational pipeline flows from market, volatility, macro, and news inputs through ingestion, features, "
        "risk scoring, and report generation, with explicit support for mock and live execution modes."
    ),
    "fig_5_3_docker_deployment_diagram.png": (
        "**Figure 5.3: Docker Deployment Diagram with CLI Entry Points and Mock/Live Modes.** "
        "The container bundles the daily pipeline and documentation generator, exposing CLI entry points for mock and live "
        "runs while keeping deployment CPU-only and reproducible."
    ),
    "fig_5_4_phase_iii_integration_validation_methodology.png": (
        "**Figure 5.4: Phase III Integration and Validation Methodology Flow.** "
        "The methodology gates progression through contract checks, integration tests, calibration, and historical backtesting "
        "before release."
    ),
}

# Combined caption dictionary
ALL_CAPTIONS = {
    **ARCHITECTURE_CAPTIONS,
    **FEATURE_CAPTIONS,
    **RISK_ENGINE_CAPTIONS,
    **EVALUATION_CAPTIONS,
    **DEMO_CAPTIONS,
    **CHAPTER_CAPTIONS,
}


def get_caption(filename: str) -> str:
    """
    Get caption for a figure file.

    Args:
        filename: Figure filename (e.g., 'mews_architecture.png')

    Returns:
        Caption string or empty string if not found
    """
    return ALL_CAPTIONS.get(filename, "")


def format_caption_for_markdown(filename: str) -> str:
    """
    Format caption for README markdown.

    Args:
        filename: Figure filename

    Returns:
        Markdown-formatted caption with italics
    """
    caption = get_caption(filename)
    if caption:
        return f"*{caption}*"
    return ""
