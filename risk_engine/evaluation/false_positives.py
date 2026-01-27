"""
False-Positive Analysis for MEWS-FIN Evaluation.

Phase 4.3: Measure false alarm burden outside crisis windows.

Key Questions:
    - How often does the system raise alerts when no crisis occurs?
    - What is the average duration of false alarms?
    - What is the worst-case (longest) false alarm streak?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .crises import CrisisWindow, get_crisis_windows, is_crisis_date
from .lead_time import EVALUATION_THRESHOLDS


@dataclass(frozen=True)
class FalseAlarmStreak:
    """
    A continuous period of false alarms.

    Attributes:
        start_date: First day of streak
        end_date: Last day of streak
        duration_days: Number of consecutive days
    """

    start_date: date
    end_date: date
    duration_days: int

    def __post_init__(self) -> None:
        if self.duration_days < 1:
            raise ValueError("Streak duration must be >= 1")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")


@dataclass
class FalsePositiveResult:
    """
    False-positive analysis for a single threshold.

    Attributes:
        threshold: Risk score threshold
        total_non_crisis_days: Total days outside crisis windows
        false_alarm_days: Days with score >= threshold outside crisis
        false_positive_rate: Fraction of non-crisis days with false alarms
        avg_streak_duration: Average false alarm streak length
        max_streak_duration: Maximum consecutive false alarm days
        n_streaks: Number of distinct false alarm periods
        streaks: List of all false alarm streaks
    """

    threshold: float
    total_non_crisis_days: int
    false_alarm_days: int
    false_positive_rate: float
    avg_streak_duration: float | None
    max_streak_duration: int | None
    n_streaks: int
    streaks: list[FalseAlarmStreak]


def find_false_alarm_streaks(
    risk_scores: dict[date, float],
    threshold: float,
    crises: Sequence[CrisisWindow] | None = None,
) -> list[FalseAlarmStreak]:
    """
    Find all consecutive false alarm streaks.

    A streak is a sequence of consecutive dates where:
        - Date is NOT in any crisis window
        - Score >= threshold

    Args:
        risk_scores: Dict of date -> risk_score
        threshold: Alert threshold
        crises: Crisis windows (uses defaults if None)

    Returns:
        List of FalseAlarmStreak objects
    """
    if crises is None:
        crises = get_crisis_windows()

    # Filter to non-crisis dates with scores
    non_crisis_alerts: list[date] = []
    for d, score in risk_scores.items():
        if not is_crisis_date(d) and score >= threshold:
            non_crisis_alerts.append(d)

    if not non_crisis_alerts:
        return []

    # Sort and find consecutive streaks
    sorted_dates = sorted(non_crisis_alerts)
    streaks: list[FalseAlarmStreak] = []

    streak_start = sorted_dates[0]
    streak_end = sorted_dates[0]

    for i in range(1, len(sorted_dates)):
        current = sorted_dates[i]
        prev = sorted_dates[i - 1]

        # Check if consecutive (allowing for weekends/holidays up to 3 days gap)
        # For strict analysis, use 1-day gap only
        if (current - prev).days == 1:
            streak_end = current
        else:
            # End current streak, start new one
            duration = (streak_end - streak_start).days + 1
            streaks.append(FalseAlarmStreak(
                start_date=streak_start,
                end_date=streak_end,
                duration_days=duration,
            ))
            streak_start = current
            streak_end = current

    # Don't forget the last streak
    duration = (streak_end - streak_start).days + 1
    streaks.append(FalseAlarmStreak(
        start_date=streak_start,
        end_date=streak_end,
        duration_days=duration,
    ))

    return streaks


def compute_false_positives(
    risk_scores: dict[date, float],
    threshold: float,
    crises: Sequence[CrisisWindow] | None = None,
) -> FalsePositiveResult:
    """
    Compute false-positive metrics for a single threshold.

    Args:
        risk_scores: Dict of date -> risk_score
        threshold: Alert threshold
        crises: Crisis windows (uses defaults if None)

    Returns:
        FalsePositiveResult with all metrics

    Raises:
        ValueError: If threshold not in [0, 1]
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

    if crises is None:
        crises = get_crisis_windows()

    # Count non-crisis days and false alarms
    total_non_crisis_days = 0
    false_alarm_days = 0

    for d, score in risk_scores.items():
        if not is_crisis_date(d):
            total_non_crisis_days += 1
            if score >= threshold:
                false_alarm_days += 1

    # Compute rate
    if total_non_crisis_days > 0:
        false_positive_rate = false_alarm_days / total_non_crisis_days
    else:
        false_positive_rate = 0.0

    # Find streaks
    streaks = find_false_alarm_streaks(risk_scores, threshold, crises)
    n_streaks = len(streaks)

    if streaks:
        durations = [s.duration_days for s in streaks]
        avg_streak_duration = sum(durations) / len(durations)
        max_streak_duration = max(durations)
    else:
        avg_streak_duration = None
        max_streak_duration = None

    return FalsePositiveResult(
        threshold=threshold,
        total_non_crisis_days=total_non_crisis_days,
        false_alarm_days=false_alarm_days,
        false_positive_rate=false_positive_rate,
        avg_streak_duration=avg_streak_duration,
        max_streak_duration=max_streak_duration,
        n_streaks=n_streaks,
        streaks=streaks,
    )


def compute_all_false_positives(
    risk_scores: dict[date, float],
    thresholds: Sequence[float] | None = None,
    crises: Sequence[CrisisWindow] | None = None,
) -> dict[float, FalsePositiveResult]:
    """
    Compute false-positive analysis for all thresholds.

    Args:
        risk_scores: Dict of date -> risk_score
        thresholds: Thresholds to evaluate (uses EVALUATION_THRESHOLDS if None)
        crises: Crisis windows (uses defaults if None)

    Returns:
        Dict of threshold -> FalsePositiveResult
    """
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS

    return {
        threshold: compute_false_positives(risk_scores, threshold, crises)
        for threshold in thresholds
    }


def false_positive_burden_score(
    result: FalsePositiveResult,
    rate_weight: float = 0.5,
    streak_weight: float = 0.5,
    max_acceptable_rate: float = 0.20,
    max_acceptable_streak: int = 30,
) -> float:
    """
    Compute a composite false-positive burden score.

    Lower is better. 0 = no burden, 1 = maximum acceptable burden.

    Args:
        result: False positive result
        rate_weight: Weight for rate component
        streak_weight: Weight for streak component
        max_acceptable_rate: Rate above which burden = 1
        max_acceptable_streak: Streak length above which burden = 1

    Returns:
        Burden score in [0, 1+] (can exceed 1 if beyond acceptable)
    """
    rate_burden = result.false_positive_rate / max_acceptable_rate

    if result.max_streak_duration is not None:
        streak_burden = result.max_streak_duration / max_acceptable_streak
    else:
        streak_burden = 0.0

    return rate_weight * rate_burden + streak_weight * streak_burden
