"""
Lead-Time Analysis for MEWS-FIN Evaluation.

Phase 4.3: Compute how early the system warns before crises.

Key Questions:
    - How many days before crisis start did risk score exceed threshold?
    - What is the mean/median/worst-case lead time?
    - Which crises were missed (no alert before start)?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .crises import CrisisWindow, get_crisis_windows

# Thresholds to evaluate (from spec - MANDATORY)
EVALUATION_THRESHOLDS: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80)


@dataclass(frozen=True)
class LeadTimeResult:
    """
    Lead-time result for a single crisis at a single threshold.

    Attributes:
        crisis_name: Name of the crisis
        threshold: Risk score threshold evaluated
        first_alert_date: First date where score >= threshold (None if missed)
        crisis_start_date: Start date of the crisis
        lead_time_days: Days between first alert and crisis start (None if missed)
        detected: Whether an alert occurred before crisis start
    """

    crisis_name: str
    threshold: float
    first_alert_date: date | None
    crisis_start_date: date
    lead_time_days: int | None
    detected: bool

    def __post_init__(self) -> None:
        """Validate lead time computation."""
        if self.detected:
            if self.first_alert_date is None or self.lead_time_days is None:
                raise ValueError(
                    "Detected crisis must have first_alert_date and lead_time_days"
                )
            if self.lead_time_days < 0:
                raise ValueError(
                    f"Lead time cannot be negative, got {self.lead_time_days}"
                )


@dataclass
class LeadTimeSummary:
    """
    Aggregated lead-time statistics across all crises for a threshold.

    Attributes:
        threshold: Risk score threshold
        results: Individual crisis results
        n_detected: Number of crises detected
        n_missed: Number of crises missed
        mean_lead_time: Mean lead time (detected crises only)
        median_lead_time: Median lead time
        min_lead_time: Worst-case (minimum) lead time
        max_lead_time: Best-case (maximum) lead time
    """

    threshold: float
    results: list[LeadTimeResult]
    n_detected: int
    n_missed: int
    mean_lead_time: float | None
    median_lead_time: float | None
    min_lead_time: int | None
    max_lead_time: int | None

    @property
    def detection_rate(self) -> float:
        """Fraction of crises detected."""
        total = self.n_detected + self.n_missed
        return self.n_detected / total if total > 0 else 0.0


def compute_lead_time(
    risk_scores: dict[date, float],
    crisis: CrisisWindow,
    threshold: float,
    lookback_days: int = 365,
) -> LeadTimeResult:
    """
    Compute lead time for a single crisis at a single threshold.

    Args:
        risk_scores: Dict of date -> risk_score
        crisis: Crisis window to evaluate
        threshold: Score threshold for alert
        lookback_days: How far back to look for first alert

    Returns:
        LeadTimeResult with detection status and lead time

    Raises:
        ValueError: If threshold not in [0, 1]
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    # Look for first alert in lookback window before crisis start
    from datetime import timedelta

    search_start = crisis.start_date - timedelta(days=lookback_days)
    first_alert_date: date | None = None

    # Sort dates and search chronologically (no lookahead)
    sorted_dates = sorted(d for d in risk_scores.keys() if search_start <= d < crisis.start_date)

    for d in sorted_dates:
        score = risk_scores[d]
        if score >= threshold:
            first_alert_date = d
            break

    if first_alert_date is not None:
        lead_time_days = (crisis.start_date - first_alert_date).days
        return LeadTimeResult(
            crisis_name=crisis.name,
            threshold=threshold,
            first_alert_date=first_alert_date,
            crisis_start_date=crisis.start_date,
            lead_time_days=lead_time_days,
            detected=True,
        )
    else:
        return LeadTimeResult(
            crisis_name=crisis.name,
            threshold=threshold,
            first_alert_date=None,
            crisis_start_date=crisis.start_date,
            lead_time_days=None,
            detected=False,
        )


def compute_lead_time_summary(
    risk_scores: dict[date, float],
    threshold: float,
    crises: Sequence[CrisisWindow] | None = None,
    lookback_days: int = 365,
) -> LeadTimeSummary:
    """
    Compute lead-time summary across all crises for a threshold.

    Args:
        risk_scores: Dict of date -> risk_score
        threshold: Score threshold for alert
        crises: Crisis windows (uses defaults if None)
        lookback_days: How far back to look for alerts

    Returns:
        LeadTimeSummary with aggregated statistics
    """
    if crises is None:
        crises = get_crisis_windows()

    results: list[LeadTimeResult] = []
    for crisis in crises:
        result = compute_lead_time(risk_scores, crisis, threshold, lookback_days)
        results.append(result)

    # Aggregate
    detected_lead_times = [r.lead_time_days for r in results if r.detected and r.lead_time_days is not None]
    n_detected = len(detected_lead_times)
    n_missed = len(results) - n_detected

    if detected_lead_times:
        mean_lead_time = sum(detected_lead_times) / len(detected_lead_times)
        sorted_times = sorted(detected_lead_times)
        n = len(sorted_times)
        if n % 2 == 0:
            median_lead_time = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        else:
            median_lead_time = float(sorted_times[n // 2])
        min_lead_time = min(detected_lead_times)
        max_lead_time = max(detected_lead_times)
    else:
        mean_lead_time = None
        median_lead_time = None
        min_lead_time = None
        max_lead_time = None

    return LeadTimeSummary(
        threshold=threshold,
        results=results,
        n_detected=n_detected,
        n_missed=n_missed,
        mean_lead_time=mean_lead_time,
        median_lead_time=median_lead_time,
        min_lead_time=min_lead_time,
        max_lead_time=max_lead_time,
    )


def compute_all_lead_times(
    risk_scores: dict[date, float],
    thresholds: Sequence[float] | None = None,
    crises: Sequence[CrisisWindow] | None = None,
) -> dict[float, LeadTimeSummary]:
    """
    Compute lead-time analysis for all thresholds.

    Args:
        risk_scores: Dict of date -> risk_score
        thresholds: Thresholds to evaluate (uses EVALUATION_THRESHOLDS if None)
        crises: Crisis windows (uses defaults if None)

    Returns:
        Dict of threshold -> LeadTimeSummary
    """
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS

    return {
        threshold: compute_lead_time_summary(risk_scores, threshold, crises)
        for threshold in thresholds
    }
