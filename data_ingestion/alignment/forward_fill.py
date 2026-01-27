"""
Forward-fill logic for macro rates.

Implements bounded forward-fill as specified in core-specs/time_alignment.yaml:
- Forward-fill max gap: 5 trading days
- Applicable to: macro_rates, volatility_indices
- After gap exceeds max, value becomes NULL

CRITICAL: Forward-fill MUST happen AFTER alignment, not in adapters.
This ensures publication lag is enforced before filling.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from .calendar import NYSECalendar


@dataclass
class ForwardFillConfig:
    """Configuration for forward-fill behavior."""

    max_gap_trading_days: int = 5
    applicable_datasets: tuple[str, ...] = ("macro_rates", "volatility_indices")

    def is_applicable(self, dataset_name: str) -> bool:
        """Check if forward-fill applies to this dataset."""
        return dataset_name in self.applicable_datasets


# Default config from time_alignment.yaml
DEFAULT_FORWARD_FILL_CONFIG = ForwardFillConfig(
    max_gap_trading_days=5,
    applicable_datasets=("macro_rates", "volatility_indices"),
)


def forward_fill_series(
    aligned_records: list[dict[str, Any]],
    series_id_field: str,
    value_field: str,
    date_field: str,
    calendar: NYSECalendar,
    config: ForwardFillConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Apply bounded forward-fill to aligned records.

    This function:
    1. Groups records by series_id
    2. Fills gaps up to max_gap_trading_days
    3. Sets value to NULL when gap exceeds limit

    Args:
        aligned_records: List of aligned record dicts with series_id, value, date
        series_id_field: Field name for series identifier (e.g., 'series_id')
        value_field: Field name for value (e.g., 'value')
        date_field: Field name for aligned date (e.g., 'aligned_to_date')
        calendar: NYSE calendar for trading day calculations
        config: Forward-fill configuration

    Returns:
        List of records with forward-fill applied (new list, does not mutate input)

    Example:
        Input (with gap):
            2024-01-15: DGS10 = 4.12
            2024-01-16: DGS10 = NULL (missing)
            2024-01-17: DGS10 = NULL (missing)
            2024-01-18: DGS10 = 4.15

        Output (filled):
            2024-01-15: DGS10 = 4.12
            2024-01-16: DGS10 = 4.12 (filled from 01-15)
            2024-01-17: DGS10 = 4.12 (filled from 01-15)
            2024-01-18: DGS10 = 4.15 (original)

        If gap > 5 trading days, value becomes NULL.
    """
    if config is None:
        config = DEFAULT_FORWARD_FILL_CONFIG

    # Group records by series_id
    by_series: dict[str, list[dict[str, Any]]] = {}
    for record in aligned_records:
        sid = record[series_id_field]
        if sid not in by_series:
            by_series[sid] = []
        by_series[sid].append(record.copy())  # Copy to avoid mutation

    result: list[dict[str, Any]] = []

    for _series_id, series_records in by_series.items():
        # Sort by date
        series_records.sort(key=lambda r: r[date_field])

        # Apply forward-fill per series
        filled = _forward_fill_single_series(
            series_records,
            value_field,
            date_field,
            calendar,
            config.max_gap_trading_days,
        )
        result.extend(filled)

    return result


def _forward_fill_single_series(
    records: list[dict[str, Any]],
    value_field: str,
    date_field: str,
    calendar: NYSECalendar,
    max_gap: int,
) -> list[dict[str, Any]]:
    """
    Apply forward-fill to a single series.

    Args:
        records: Records for one series, sorted by date
        value_field: Field name for value
        date_field: Field name for date
        calendar: NYSE calendar
        max_gap: Maximum gap in trading days

    Returns:
        Records with forward-fill applied
    """
    if not records:
        return records

    last_known_value: Any = None
    last_known_date: date | None = None
    result: list[dict[str, Any]] = []

    for record in records:
        current_date = record[date_field]
        current_value = record.get(value_field)

        if current_value is not None:
            # We have a value - update last known
            last_known_value = current_value
            last_known_date = current_date
            result.append(record)
        else:
            # Value is None - check if we can forward-fill
            if last_known_value is not None and last_known_date is not None:
                # Count trading days since last known value
                gap_days = _count_trading_days_between(
                    last_known_date, current_date, calendar
                )

                if gap_days <= max_gap:
                    # Within limit - forward-fill
                    filled_record = record.copy()
                    filled_record[value_field] = last_known_value
                    filled_record["_forward_filled"] = True
                    filled_record["_filled_from_date"] = last_known_date.isoformat()
                    filled_record["_gap_trading_days"] = gap_days
                    result.append(filled_record)
                else:
                    # Gap exceeded - keep as NULL
                    null_record = record.copy()
                    null_record[value_field] = None
                    null_record["_fill_expired"] = True
                    null_record["_gap_trading_days"] = gap_days
                    result.append(null_record)
            else:
                # No previous value to fill from
                result.append(record)

    return result


def _count_trading_days_between(
    start_date: date,
    end_date: date,
    calendar: NYSECalendar,
) -> int:
    """
    Count trading days between two dates (exclusive of start, inclusive of end).

    Args:
        start_date: Start date (the date we have a value for)
        end_date: End date (the date we're trying to fill)
        calendar: NYSE calendar

    Returns:
        Number of trading days in the gap
    """
    if end_date <= start_date:
        return 0

    count = 0
    current = start_date
    from datetime import timedelta

    while current < end_date:
        current = current + timedelta(days=1)
        if calendar.is_trading_day(current):
            count += 1

    return count


def generate_missing_dates(
    existing_records: list[dict[str, Any]],
    series_id_field: str,
    date_field: str,
    start_date: date,
    end_date: date,
    calendar: NYSECalendar,
) -> list[dict[str, Any]]:
    """
    Generate placeholder records for missing trading dates.

    This is used to ensure we have records for all trading days
    so forward-fill can be applied consistently.

    Args:
        existing_records: Records that exist
        series_id_field: Field name for series_id
        date_field: Field name for date
        start_date: Start of date range
        end_date: End of date range
        calendar: NYSE calendar

    Returns:
        List of placeholder records for missing dates
    """
    # Get all series IDs
    series_ids = {r[series_id_field] for r in existing_records}

    # Build set of (series_id, date) that exist
    existing = {
        (r[series_id_field], r[date_field])
        for r in existing_records
    }

    # Get all trading days in range
    trading_days = calendar.get_trading_days(start_date, end_date)

    # Generate missing records
    missing: list[dict[str, Any]] = []

    for series_id in series_ids:
        for trading_date in trading_days:
            if (series_id, trading_date) not in existing:
                missing.append({
                    series_id_field: series_id,
                    date_field: trading_date,
                    "value": None,  # Placeholder for forward-fill
                    "_generated": True,
                })

    return missing
