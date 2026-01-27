"""
Report Generation for Daily Pipeline.

MEWS-FIN Phase 5A: Generate human-readable daily risk reports.

Text-only output. Deterministic. Designed for terminal and documentation.
"""

from __future__ import annotations

from .context import PipelineContext


def generate_report(ctx: PipelineContext) -> str:
    """
    Generate complete daily risk report.

    Args:
        ctx: Completed pipeline context

    Returns:
        Formatted report string
    """
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append("=" * 70)
    lines.append(f" MEWS-FIN Daily Risk Report — {ctx.run_date.isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    # Overall score and regime
    if ctx.risk.final_score is not None:
        score_str = f"{ctx.risk.final_score:.2f}"
        lines.append(f"Overall Risk Score: {score_str} ({ctx.risk.regime})")
    else:
        lines.append("Overall Risk Score: UNAVAILABLE")
    lines.append("")

    # Sub-scores section
    if ctx.config.report.show_sub_scores and ctx.risk.sub_scores:
        lines.append("Sub-scores:")
        for name, value in sorted(ctx.risk.sub_scores.items()):
            if value is not None:
                # Format name nicely
                display_name = name.replace("_", " ").title()
                lines.append(f"  - {display_name}: {value:.2f}")
        lines.append("")

    # Dominant drivers section
    if ctx.config.report.show_dominant_factors and ctx.risk.dominant_factors:
        lines.append("Dominant Drivers:")
        for i, (factor, contribution) in enumerate(
            ctx.risk.dominant_factors[: ctx.config.report.top_factors_count], 1
        ):
            sign = "+" if contribution >= 0 else ""
            # Format factor name nicely
            display_name = factor.replace("_", " ").replace("20d", "(20d)").replace("5d", "(5d)")
            lines.append(f"  {i}. {display_name} ({sign}{contribution:.2f})")
        lines.append("")

    # Model breakdown section
    if ctx.config.report.show_model_breakdown:
        lines.append("Model Breakdown:")
        if ctx.risk.heuristic_score is not None:
            lines.append(f"  - Heuristic: {ctx.risk.heuristic_score:.2f}")
        if ctx.risk.ml_scores:
            for model, score in ctx.risk.ml_scores.items():
                model_name = model.replace("_", " ").title()
                lines.append(f"  - ML ({model_name}): {score:.2f}")
        if ctx.risk.ensemble_score is not None:
            lines.append(f"  - Ensemble (calibrated): {ctx.risk.ensemble_score:.2f}")
        lines.append("")

    # Regime rationale section
    if ctx.config.report.show_regime_rationale and ctx.risk.regime_rationale:
        lines.append("Regime Rationale:")
        # Wrap text at ~65 chars
        rationale = ctx.risk.regime_rationale
        wrapped = _wrap_text(rationale, width=65, indent="  ")
        lines.append(wrapped)
        lines.append("")

    # Run metadata
    lines.append("-" * 70)
    lines.append(f"Run ID: {ctx.run_id}")
    lines.append(f"As-of: {ctx.as_of.isoformat()}")
    if ctx.duration_seconds:
        lines.append(f"Duration: {ctx.duration_seconds:.2f}s")
    if ctx.config.mock_mode:
        lines.append("Mode: MOCK (synthetic data)")
    if ctx.warnings:
        lines.append(f"Warnings: {len(ctx.warnings)}")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def print_report(ctx: PipelineContext) -> None:
    """Print report to stdout."""
    print(generate_report(ctx))


def save_report(ctx: PipelineContext, path: str) -> None:
    """Save report to file."""
    report = generate_report(ctx)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def generate_summary_line(ctx: PipelineContext) -> str:
    """
    Generate single-line summary.

    Useful for logging and monitoring.
    """
    if ctx.risk.final_score is not None:
        return (
            f"[MEWS-FIN] {ctx.run_date} | "
            f"Score: {ctx.risk.final_score:.2f} | "
            f"Regime: {ctx.risk.regime} | "
            f"Duration: {ctx.duration_seconds:.2f}s"
        )
    else:
        return f"[MEWS-FIN] {ctx.run_date} | FAILED | Errors: {len(ctx.errors)}"


def _wrap_text(text: str, width: int = 65, indent: str = "") -> str:
    """Wrap text to specified width with indent."""
    words = text.split()
    lines = []
    current_line = indent

    for word in words:
        if len(current_line) + len(word) + 1 <= width + len(indent):
            if current_line == indent:
                current_line += word
            else:
                current_line += " " + word
        else:
            lines.append(current_line)
            current_line = indent + word

    if current_line.strip():
        lines.append(current_line)

    return "\n".join(lines)
