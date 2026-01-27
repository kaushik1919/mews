"""
Crisis Window Definitions for MEWS-FIN Evaluation.

Phase 4.3: Ground truth crisis windows for lead-time analysis.

These windows are evaluation truth, NOT labels to optimize.
Aligned with core-specs/risk_score.yaml historical anchors.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CrisisWindow:
    """
    Defines a historical crisis period for evaluation.

    Attributes:
        name: Human-readable crisis name
        start_date: First day of crisis (when stress became evident)
        peak_date: Date of maximum stress
        end_date: Last day of crisis period
        expected_score_range: Expected risk score range at peak
        description: Brief description for reporting
    """

    name: str
    start_date: date
    peak_date: date
    end_date: date
    expected_score_range: tuple[float, float]
    description: str

    def __post_init__(self) -> None:
        """Validate crisis window integrity."""
        if self.start_date > self.peak_date:
            raise ValueError(
                f"Crisis '{self.name}': start_date must be <= peak_date"
            )
        if self.peak_date > self.end_date:
            raise ValueError(
                f"Crisis '{self.name}': peak_date must be <= end_date"
            )
        if not (0.0 <= self.expected_score_range[0] <= 1.0):
            raise ValueError(
                f"Crisis '{self.name}': invalid expected_score_range lower bound"
            )
        if not (0.0 <= self.expected_score_range[1] <= 1.0):
            raise ValueError(
                f"Crisis '{self.name}': invalid expected_score_range upper bound"
            )

    def contains(self, d: date) -> bool:
        """Check if a date falls within this crisis window."""
        return self.start_date <= d <= self.end_date

    def days_before_start(self, d: date) -> int | None:
        """
        Compute days before crisis start.

        Returns:
            Number of days before start_date if d < start_date, else None
        """
        if d < self.start_date:
            return (self.start_date - d).days
        return None


# ==============================================================================
# GROUND TRUTH CRISIS WINDOWS
# ==============================================================================
# These are fixed evaluation windows aligned with risk_score.yaml anchors.
# Do NOT modify these without updating the specification.

CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow(
        name="2008 Global Financial Crisis",
        start_date=date(2008, 9, 15),  # Lehman bankruptcy
        peak_date=date(2008, 10, 10),  # VIX peak ~80
        end_date=date(2009, 3, 9),  # Market bottom
        expected_score_range=(0.85, 1.0),
        description="Maximum systemic stress in modern history",
    ),
    CrisisWindow(
        name="2011 Eurozone Crisis",
        start_date=date(2011, 7, 1),  # Contagion from Greece
        peak_date=date(2011, 8, 8),  # S&P downgrade aftermath
        end_date=date(2011, 10, 4),  # ECB intervention
        expected_score_range=(0.60, 0.80),
        description="European sovereign debt contagion",
    ),
    CrisisWindow(
        name="2020 COVID Crash",
        start_date=date(2020, 2, 20),  # First major selloff
        peak_date=date(2020, 3, 16),  # Circuit breaker day
        end_date=date(2020, 3, 23),  # Fed intervention bottom
        expected_score_range=(0.80, 0.95),
        description="Rapid onset pandemic-driven liquidity crisis",
    ),
    CrisisWindow(
        name="2022 Rate-Hike Drawdown",
        start_date=date(2022, 1, 3),  # Year open, Fed pivot
        peak_date=date(2022, 6, 16),  # FOMC 75bp hike
        end_date=date(2022, 10, 12),  # Inflation peak
        expected_score_range=(0.50, 0.75),
        description="Sustained stress from aggressive monetary tightening",
    ),
)


def get_crisis_windows() -> tuple[CrisisWindow, ...]:
    """Return all ground truth crisis windows."""
    return CRISIS_WINDOWS


def is_crisis_date(d: date) -> bool:
    """Check if a date falls within any crisis window."""
    return any(crisis.contains(d) for crisis in CRISIS_WINDOWS)


def get_crisis_for_date(d: date) -> CrisisWindow | None:
    """Return the crisis window containing a date, or None."""
    for crisis in CRISIS_WINDOWS:
        if crisis.contains(d):
            return crisis
    return None


def iter_non_crisis_dates(
    start: date,
    end: date,
) -> Iterator[date]:
    """
    Iterate over dates NOT in any crisis window.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Yields:
        Dates not contained in any crisis window
    """
    from datetime import timedelta

    current = start
    while current <= end:
        if not is_crisis_date(current):
            yield current
        current += timedelta(days=1)


def get_evaluation_date_range() -> tuple[date, date]:
    """
    Get the full date range covered by crisis windows.

    Returns reasonable bounds for evaluation data.
    """
    min_date = min(c.start_date for c in CRISIS_WINDOWS)
    max_date = max(c.end_date for c in CRISIS_WINDOWS)
    # Extend 1 year before first crisis for lead-time analysis
    from datetime import timedelta
    return (min_date - timedelta(days=365), max_date)
