"""
MEWS-FIN Daily Pipeline Entry Point.

Phase 5A: Orchestrated daily risk pipeline.

Usage:
    python -m pipeline.daily_run.run --date 2024-01-15
    python -m pipeline.daily_run.run --mock
    python -m pipeline.daily_run.run --help

This is the main entry point for running the complete MEWS-FIN pipeline.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

from .config import DEFAULT_CONFIG, MOCK_CONFIG
from .context import PipelineContext, create_context
from .features import run_features
from .ingestion import run_ingestion
from .report import generate_summary_line, print_report
from .risk import run_risk


def run_pipeline(ctx: PipelineContext) -> PipelineContext:
    """
    Execute complete daily pipeline.

    Steps (in order):
        1. Ingestion - load data from all sources
        2. Features - compute numeric, sentiment, graph features
        3. Risk - compute heuristic, ML, ensemble scores
        4. Report - generate human-readable output

    Args:
        ctx: Pipeline context (created via create_context)

    Returns:
        Completed context with all results
    """
    # Step 1: Ingestion
    if ctx.verbose:
        print("[MEWS] Running ingestion...")
    run_ingestion(ctx)

    # Step 2: Features
    if ctx.verbose:
        print("[MEWS] Computing features...")
    run_features(ctx)

    # Step 3: Risk
    if ctx.verbose:
        print("[MEWS] Scoring risk...")
    run_risk(ctx)

    # Mark complete
    ctx.completed_at = datetime.now(timezone.utc)
    if ctx.verbose:
        print(f"[MEWS] Pipeline complete in {ctx.duration_seconds:.2f}s")

    return ctx


def main(args: list[str] | None = None) -> int:
    """
    CLI entry point.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        prog="mews-daily",
        description="MEWS-FIN Daily Risk Pipeline",
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date in YYYY-MM-DD format (default: today)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode with synthetic data",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose execution output",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary line, not full report",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file",
    )

    parsed = parser.parse_args(args)

    # Parse date
    if parsed.date:
        try:
            run_date = date.fromisoformat(parsed.date)
        except ValueError:
            print(f"Error: Invalid date format: {parsed.date}")
            print("Use YYYY-MM-DD format (e.g., 2024-01-15)")
            return 1
    else:
        run_date = date.today()

    # Select config
    if parsed.mock:
        config = MOCK_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Resolve verbose flag (--quiet overrides --verbose)
    if parsed.quiet:
        verbose = False
    else:
        verbose = parsed.verbose

    # Create context
    ctx = create_context(run_date=run_date, config=config, verbose=verbose)

    try:
        # Run pipeline
        ctx = run_pipeline(ctx)

        # Output
        if parsed.quiet:
            print(generate_summary_line(ctx))
        else:
            print_report(ctx)

        # Save to file if requested
        if parsed.output:
            from .report import save_report
            save_report(ctx, parsed.output)
            print(f"Report saved to: {parsed.output}")

        # Return success if we got a score
        if ctx.is_success:
            return 0
        else:
            print(f"Pipeline completed with errors: {ctx.errors}")
            return 1

    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
