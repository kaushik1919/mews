"""
Pipeline Execution Context.

MEWS-FIN Phase 5A: Shared state container for a single pipeline run.

The context is created once at pipeline start and passed through all steps.
It carries timestamps, paths, intermediate results, and timing information.

No global state. Everything is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from .config import PipelineConfig


@dataclass
class StepTiming:
    """Timing information for a pipeline step."""

    step_name: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    success: bool = False
    error_message: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Duration in seconds, or None if not complete."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class IngestionResult:
    """Results from ingestion step."""

    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)
    sources_loaded: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all sources loaded successfully."""
        return len(self.sources_failed) == 0 and len(self.sources_loaded) > 0


@dataclass
class FeatureResult:
    """Results from feature computation step."""

    numeric_features: dict[str, float | None] = field(default_factory=dict)
    sentiment_features: dict[str, float | None] = field(default_factory=dict)
    graph_features: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_features(self) -> dict[str, float | None]:
        """Merge all features into single dict."""
        return {
            **self.numeric_features,
            **self.sentiment_features,
            **self.graph_features,
        }

    @property
    def is_complete(self) -> bool:
        """Check if all features have values."""
        return all(v is not None for v in self.all_features.values())


@dataclass
class RiskResult:
    """Results from risk scoring step."""

    # Individual model scores
    heuristic_score: float | None = None
    ml_scores: dict[str, float] = field(default_factory=dict)
    ensemble_score: float | None = None

    # Final output
    final_score: float | None = None
    regime: str = "UNKNOWN"

    # Sub-scores
    sub_scores: dict[str, float | None] = field(default_factory=dict)

    # Explainability
    dominant_factors: list[tuple[str, float]] = field(default_factory=list)
    regime_rationale: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """
    Complete context for a single pipeline run.

    Created at pipeline start, passed through all steps.
    Accumulates results and timing information.
    """

    # Run identification
    run_id: str
    run_date: date
    as_of: pd.Timestamp

    # Configuration
    config: PipelineConfig

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    step_timings: list[StepTiming] = field(default_factory=list)

    # Results from each step
    ingestion: IngestionResult = field(default_factory=IngestionResult)
    features: FeatureResult = field(default_factory=FeatureResult)
    risk: RiskResult = field(default_factory=RiskResult)

    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if pipeline completed successfully."""
        return len(self.errors) == 0 and self.risk.final_score is not None

    @property
    def duration_seconds(self) -> float | None:
        """Total duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def start_step(self, step_name: str) -> StepTiming:
        """Record start of a pipeline step."""
        timing = StepTiming(
            step_name=step_name,
            started_at=datetime.now(timezone.utc),
        )
        self.step_timings.append(timing)
        return timing

    def complete_step(self, timing: StepTiming, success: bool = True, error: str | None = None) -> None:
        """Record completion of a pipeline step."""
        timing.completed_at = datetime.now(timezone.utc)
        timing.success = success
        timing.error_message = error
        if error:
            self.errors.append(f"[{timing.step_name}] {error}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_date": self.run_date.isoformat(),
            "as_of": self.as_of.isoformat(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_success": self.is_success,
            "duration_seconds": self.duration_seconds,
            "config": self.config.to_dict(),
            "warnings": self.warnings,
            "errors": self.errors,
        }


def create_context(
    run_date: date | None = None,
    as_of: pd.Timestamp | None = None,
    config: PipelineConfig | None = None,
) -> PipelineContext:
    """
    Create a new pipeline context.

    Args:
        run_date: Date for the run (defaults to today)
        as_of: Point-in-time timestamp (defaults to run_date 21:00 UTC)
        config: Pipeline configuration (defaults to DEFAULT_CONFIG)

    Returns:
        New PipelineContext ready for execution
    """
    from .config import DEFAULT_CONFIG

    if run_date is None:
        run_date = date.today()

    if as_of is None:
        # Default to 21:00 UTC (after US market close)
        as_of = pd.Timestamp(
            year=run_date.year,
            month=run_date.month,
            day=run_date.day,
            hour=21,
            minute=0,
            second=0,
            tz="UTC",
        )

    if config is None:
        config = DEFAULT_CONFIG

    # Generate run ID
    run_id = f"mews-{run_date.isoformat()}-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    return PipelineContext(
        run_id=run_id,
        run_date=run_date,
        as_of=as_of,
        config=config,
    )
