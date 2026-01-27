"""
Evaluation Report Generation for MEWS-FIN.

Phase 4.3: Generate human-readable text reports.

Output format is plain text with ASCII tables.
Designed for terminal output and documentation.
"""

from __future__ import annotations

from .compare import ModelComparison
from .false_positives import FalsePositiveResult
from .lead_time import LeadTimeSummary


def format_lead_time_table(
    lead_times: dict[float, LeadTimeSummary],
    title: str = "Lead-Time Analysis",
) -> str:
    """
    Format lead-time results as ASCII table.

    Args:
        lead_times: Lead time summaries by threshold
        title: Table title

    Returns:
        Formatted ASCII table
    """
    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" {title}")
    lines.append(f"{'=' * 80}")

    # Header
    header = f"{'Threshold':>10} | {'Detected':>8} | {'Missed':>6} | {'Mean':>8} | {'Median':>8} | {'Min':>6} | {'Max':>6}"
    lines.append(header)
    lines.append("-" * 80)

    for threshold in sorted(lead_times.keys()):
        lt = lead_times[threshold]
        mean_str = f"{lt.mean_lead_time:.1f}d" if lt.mean_lead_time is not None else "N/A"
        median_str = f"{lt.median_lead_time:.1f}d" if lt.median_lead_time is not None else "N/A"
        min_str = f"{lt.min_lead_time}d" if lt.min_lead_time is not None else "N/A"
        max_str = f"{lt.max_lead_time}d" if lt.max_lead_time is not None else "N/A"

        row = (
            f"{threshold:>10.2f} | "
            f"{lt.n_detected:>8} | "
            f"{lt.n_missed:>6} | "
            f"{mean_str:>8} | "
            f"{median_str:>8} | "
            f"{min_str:>6} | "
            f"{max_str:>6}"
        )
        lines.append(row)

    return "\n".join(lines)


def format_crisis_detail_table(
    lead_times: dict[float, LeadTimeSummary],
    threshold: float,
    title: str = "Crisis-by-Crisis Lead Time",
) -> str:
    """
    Format per-crisis lead time details.

    Args:
        lead_times: Lead time summaries
        threshold: Threshold to display
        title: Table title

    Returns:
        Formatted ASCII table
    """
    if threshold not in lead_times:
        return f"No data for threshold {threshold}"

    lt = lead_times[threshold]
    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" {title} (threshold = {threshold:.2f})")
    lines.append(f"{'=' * 80}")

    header = f"{'Crisis':40} | {'Alert Date':>12} | {'Lead Time':>10} | {'Status':>8}"
    lines.append(header)
    lines.append("-" * 80)

    for result in lt.results:
        alert_str = result.first_alert_date.isoformat() if result.first_alert_date else "N/A"
        lead_str = f"{result.lead_time_days}d" if result.lead_time_days is not None else "N/A"
        status = "DETECTED" if result.detected else "MISSED"

        row = f"{result.crisis_name:40} | {alert_str:>12} | {lead_str:>10} | {status:>8}"
        lines.append(row)

    return "\n".join(lines)


def format_false_positive_table(
    false_positives: dict[float, FalsePositiveResult],
    title: str = "False-Positive Analysis",
) -> str:
    """
    Format false-positive results as ASCII table.

    Args:
        false_positives: False positive results by threshold
        title: Table title

    Returns:
        Formatted ASCII table
    """
    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" {title}")
    lines.append(f"{'=' * 80}")

    header = (
        f"{'Threshold':>10} | {'FP Days':>8} | {'Total Days':>10} | "
        f"{'FP Rate':>8} | {'Avg Streak':>10} | {'Max Streak':>10}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for threshold in sorted(false_positives.keys()):
        fp = false_positives[threshold]
        rate_str = f"{fp.false_positive_rate * 100:.1f}%"
        avg_str = f"{fp.avg_streak_duration:.1f}d" if fp.avg_streak_duration is not None else "N/A"
        max_str = f"{fp.max_streak_duration}d" if fp.max_streak_duration is not None else "N/A"

        row = (
            f"{threshold:>10.2f} | "
            f"{fp.false_alarm_days:>8} | "
            f"{fp.total_non_crisis_days:>10} | "
            f"{rate_str:>8} | "
            f"{avg_str:>10} | "
            f"{max_str:>10}"
        )
        lines.append(row)

    return "\n".join(lines)


def format_model_comparison_table(
    comparison: ModelComparison,
    threshold: float | None = None,
) -> str:
    """
    Format model comparison as ASCII table.

    Args:
        comparison: Model comparison results
        threshold: Specific threshold to display (uses best if None)

    Returns:
        Formatted ASCII table
    """
    if threshold is None:
        threshold = comparison.recommended_threshold

    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" Model Comparison (threshold = {threshold:.2f})")
    lines.append(f"{'=' * 80}")

    header = (
        f"{'Model':25} | {'Median LT':>10} | {'Detection':>10} | "
        f"{'FP Rate':>8} | {'Score':>8}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for model_key in ["heuristic", "ml", "ensemble"]:
        model = comparison.models[model_key]
        lt = model.lead_times[threshold]
        fp = model.false_positives[threshold]

        median_str = f"{lt.median_lead_time:.0f}d" if lt.median_lead_time else "N/A"
        detect_str = f"{lt.detection_rate * 100:.0f}%"
        fp_str = f"{fp.false_positive_rate * 100:.1f}%"

        # Compute score at this threshold
        from .compare import compute_model_score
        score = compute_model_score(lt, fp)

        best_marker = " *" if model_key == comparison.best_model else ""
        row = (
            f"{model.model_name:25} | "
            f"{median_str:>10} | "
            f"{detect_str:>10} | "
            f"{fp_str:>8} | "
            f"{score:>8.3f}{best_marker}"
        )
        lines.append(row)

    lines.append("-" * 80)
    lines.append("* = Best performing model")

    return "\n".join(lines)


def format_recommendation(comparison: ModelComparison) -> str:
    """
    Format threshold recommendation.

    Args:
        comparison: Model comparison results

    Returns:
        Formatted recommendation text
    """
    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(" RECOMMENDATION")
    lines.append(f"{'=' * 80}")

    best_model = comparison.models[comparison.best_model]
    threshold = comparison.recommended_threshold
    lt = best_model.lead_times[threshold]
    fp = best_model.false_positives[threshold]

    lines.append(f"\nBest Model: {best_model.model_name}")
    lines.append(f"Recommended Threshold: {threshold:.2f}")
    lines.append("")
    lines.append("Performance at recommended threshold:")
    lines.append(f"  - Median Lead Time: {lt.median_lead_time:.0f} days" if lt.median_lead_time else "  - Median Lead Time: N/A")
    lines.append(f"  - Detection Rate: {lt.detection_rate * 100:.0f}% ({lt.n_detected}/{lt.n_detected + lt.n_missed} crises)")
    lines.append(f"  - False Positive Rate: {fp.false_positive_rate * 100:.1f}%")
    lines.append(f"  - Max False Alarm Streak: {fp.max_streak_duration} days" if fp.max_streak_duration else "  - Max False Alarm Streak: N/A")

    return "\n".join(lines)


def format_full_report(
    comparison: ModelComparison,
    show_all_thresholds: bool = True,
) -> str:
    """
    Generate complete evaluation report.

    Args:
        comparison: Model comparison results
        show_all_thresholds: Whether to show all threshold details

    Returns:
        Complete formatted report
    """
    sections: list[str] = []

    # Title
    sections.append("\n" + "=" * 80)
    sections.append(" MEWS-FIN PHASE 4.3: LEAD-TIME AND FALSE-POSITIVE ANALYSIS")
    sections.append("=" * 80)

    # Summary
    sections.append("\n" + "-" * 80)
    sections.append(" EXECUTIVE SUMMARY")
    sections.append("-" * 80)
    sections.append(f"\n{comparison.summary}")

    # Model comparison at recommended threshold
    sections.append(format_model_comparison_table(comparison))

    # Best model details
    best_model = comparison.models[comparison.best_model]

    # Lead time table
    sections.append(format_lead_time_table(
        best_model.lead_times,
        f"Lead-Time Analysis: {best_model.model_name}"
    ))

    # Crisis detail at recommended threshold
    sections.append(format_crisis_detail_table(
        best_model.lead_times,
        comparison.recommended_threshold,
    ))

    # False positive table
    sections.append(format_false_positive_table(
        best_model.false_positives,
        f"False-Positive Analysis: {best_model.model_name}"
    ))

    # Recommendation
    sections.append(format_recommendation(comparison))

    # Final conclusion
    sections.append("\n" + "=" * 80)
    sections.append(" CONCLUSION")
    sections.append("=" * 80)
    sections.append(f"\n{comparison.summary}")
    sections.append("")

    return "\n".join(sections)


def print_report(comparison: ModelComparison) -> None:
    """
    Print full evaluation report to stdout.

    Args:
        comparison: Model comparison results
    """
    print(format_full_report(comparison))


def generate_report_file(
    comparison: ModelComparison,
    output_path: str,
) -> None:
    """
    Write evaluation report to file.

    Args:
        comparison: Model comparison results
        output_path: Path for output file
    """
    report = format_full_report(comparison)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
