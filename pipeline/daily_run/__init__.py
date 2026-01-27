"""
MEWS-FIN Daily Pipeline Module.

Phase 5A: Systemization, Orchestration & Demo Pipeline.

This module orchestrates the complete MEWS-FIN daily run:
    1. Ingestion - load market data, volatility, macro, news
    2. Features - compute numeric, sentiment, graph features
    3. Risk - compute heuristic, ML, ensemble scores
    4. Report - generate human-readable output

Usage:
    # CLI
    python -m pipeline.daily_run.run --date 2024-01-15 --mock

    # Programmatic
    from pipeline.daily_run import run_pipeline, create_context

    ctx = create_context(run_date=date(2024, 1, 15), mock_mode=True)
    ctx = run_pipeline(ctx)
    print(ctx.risk.final_score)
"""

from __future__ import annotations

# Configuration
from .config import (
    DEFAULT_CONFIG,
    MOCK_CONFIG,
    FeatureConfig,
    IngestionConfig,
    PipelineConfig,
    ReportConfig,
    RiskConfig,
)

# Context
from .context import (
    FeatureResult,
    IngestionResult,
    PipelineContext,
    RiskResult,
    StepTiming,
    create_context,
)

# Pipeline steps
from .features import run_features
from .ingestion import run_ingestion
from .report import (
    generate_report,
    generate_summary_line,
    print_report,
    save_report,
)
from .risk import run_risk

# Main entry point
from .run import main, run_pipeline

__all__ = [
    # Configuration
    "PipelineConfig",
    "IngestionConfig",
    "FeatureConfig",
    "RiskConfig",
    "ReportConfig",
    "DEFAULT_CONFIG",
    "MOCK_CONFIG",
    # Context
    "PipelineContext",
    "StepTiming",
    "IngestionResult",
    "FeatureResult",
    "RiskResult",
    "create_context",
    # Pipeline steps
    "run_ingestion",
    "run_features",
    "run_risk",
    # Report
    "generate_report",
    "generate_summary_line",
    "print_report",
    "save_report",
    # Main
    "run_pipeline",
    "main",
]
