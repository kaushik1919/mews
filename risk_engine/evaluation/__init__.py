"""
MEWS-FIN Evaluation Module.

Phase 4.3: Lead-Time, Crisis Detection & False-Positive Analysis.

This module answers three critical questions:
    1. How early does MEWS-FIN warn before real crises?
    2. How often does it issue false alarms?
    3. Does the ensemble outperform heuristic and ML alone?

Usage:
    from risk_engine.evaluation import (
        compare_models,
        print_report,
        CRISIS_WINDOWS,
        EVALUATION_THRESHOLDS,
    )

    comparison = compare_models(
        heuristic_scores=heuristic_scores,
        ml_scores=ml_scores,
        ensemble_scores=ensemble_scores,
    )
    print_report(comparison)
"""

from __future__ import annotations

# Model comparison
from .compare import (
    ModelComparison,
    ModelEvaluation,
    compare_models,
    compute_model_score,
    evaluate_model,
    quick_compare,
)

# Crisis definitions
from .crises import (
    CRISIS_WINDOWS,
    CrisisWindow,
    get_crisis_for_date,
    get_crisis_windows,
    get_evaluation_date_range,
    is_crisis_date,
    iter_non_crisis_dates,
)

# False-positive analysis
from .false_positives import (
    FalseAlarmStreak,
    FalsePositiveResult,
    compute_all_false_positives,
    compute_false_positives,
    false_positive_burden_score,
    find_false_alarm_streaks,
)

# Lead-time analysis
from .lead_time import (
    EVALUATION_THRESHOLDS,
    LeadTimeResult,
    LeadTimeSummary,
    compute_all_lead_times,
    compute_lead_time,
    compute_lead_time_summary,
)

# Report generation
from .report import (
    format_crisis_detail_table,
    format_false_positive_table,
    format_full_report,
    format_lead_time_table,
    format_model_comparison_table,
    format_recommendation,
    generate_report_file,
    print_report,
)

__all__ = [
    # Crisis definitions
    "CrisisWindow",
    "CRISIS_WINDOWS",
    "get_crisis_windows",
    "is_crisis_date",
    "get_crisis_for_date",
    "iter_non_crisis_dates",
    "get_evaluation_date_range",
    # Lead-time analysis
    "EVALUATION_THRESHOLDS",
    "LeadTimeResult",
    "LeadTimeSummary",
    "compute_lead_time",
    "compute_lead_time_summary",
    "compute_all_lead_times",
    # False-positive analysis
    "FalseAlarmStreak",
    "FalsePositiveResult",
    "find_false_alarm_streaks",
    "compute_false_positives",
    "compute_all_false_positives",
    "false_positive_burden_score",
    # Model comparison
    "ModelEvaluation",
    "ModelComparison",
    "compute_model_score",
    "evaluate_model",
    "compare_models",
    "quick_compare",
    # Report generation
    "format_lead_time_table",
    "format_crisis_detail_table",
    "format_false_positive_table",
    "format_model_comparison_table",
    "format_recommendation",
    "format_full_report",
    "print_report",
    "generate_report_file",
]
