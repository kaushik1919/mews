"""
Time alignment engine.

This is the core alignment module that transforms raw records into
spec-compliant, time-aligned records.

Implements rules from core-specs/time_alignment.yaml:
- All timestamps in UTC
- Align to NYSE market close (21:00 UTC / 4:00 PM ET)
- Respect publication lags
- Prevent lookahead bias

CRITICAL: All alignment logic MUST live here, not in adapters.
"""

# Use relative import within the package
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from adapters import RawRecord
from alignment.calendar import UTC, NYSECalendar
from alignment.lag_rules import DatasetType, get_publication_lag


@dataclass
class AlignedRecord:
    """
    A time-aligned record ready for schema validation and storage.

    This is the output of the alignment process.
    - timestamp is in UTC
    - aligned_to_date is the trading date this record is assigned to
    """
    timestamp: datetime  # UTC, aligned to market close
    aligned_to_date: date  # The trading date
    asset_id: str
    data: dict[str, Any]
    source: str
    ingestion_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate aligned record."""
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        if str(self.timestamp.tzinfo) != "UTC":
            raise ValueError(f"timestamp must be UTC, got {self.timestamp.tzinfo}")


class TimeAligner:
    """
    Time alignment engine.

    Transforms raw records into aligned records following
    core-specs/time_alignment.yaml rules.

    From time_alignment.yaml:
    - global.timezone: UTC
    - market_calendar.close: 21:00:00Z
    - market_calendar.alignment_point: close
    - precision.rounding.rule: truncate_to_day
    """

    def __init__(self, calendar: NYSECalendar | None = None):
        """
        Initialize aligner.

        Args:
            calendar: NYSE calendar instance. If None, creates default.
        """
        self._calendar = calendar or NYSECalendar()

    def align_record(
        self,
        raw: RawRecord,
        dataset_type: DatasetType,
    ) -> AlignedRecord | None:
        """
        Align a single raw record to UTC market close.

        Args:
            raw: Raw record from adapter
            dataset_type: Type of dataset for lag rules

        Returns:
            AlignedRecord with UTC timestamp, or None if record
            should be skipped (e.g., non-trading day)
        """
        # Get alignment rule for this dataset
        lag_rule = get_publication_lag(dataset_type)

        # Convert raw timestamp to UTC
        raw_ts = raw.raw_timestamp
        if raw_ts.tzinfo is None:
            raise ValueError(
                f"Raw timestamp must be timezone-aware: {raw_ts}. "
                "Adapters must preserve source timezone."
            )

        raw_ts_utc = raw_ts.astimezone(UTC)

        # Apply alignment based on dataset type
        if lag_rule.alignment_mode == "native":
            # market_prices, volatility_indices
            # Already at market close, just convert to UTC
            aligned = self._align_native(raw_ts_utc)
        elif lag_rule.alignment_mode == "assign_to_previous_close":
            # macro_rates
            aligned = self._align_to_previous_close(raw_ts_utc)
        elif lag_rule.alignment_mode == "assign_to_same_day_close":
            # financial_news
            aligned = self._align_to_same_day_close(raw_ts_utc)
        elif lag_rule.alignment_mode == "same_as_news":
            # sentiment
            aligned = self._align_to_same_day_close(raw_ts_utc)
        else:
            raise ValueError(f"Unknown alignment mode: {lag_rule.alignment_mode}")

        if aligned is None:
            return None

        aligned_ts, aligned_date = aligned

        return AlignedRecord(
            timestamp=aligned_ts,
            aligned_to_date=aligned_date,
            asset_id=raw.asset_id,
            data=raw.data,
            source=raw.source,
            ingestion_metadata={
                "raw_timestamp": raw_ts.isoformat(),
                "alignment_mode": lag_rule.alignment_mode,
                "aligned_at": datetime.now(UTC).isoformat(),
            },
        )

    def align_records(
        self,
        records: list[RawRecord],
        dataset_type: DatasetType,
    ) -> list[AlignedRecord]:
        """
        Align multiple raw records.

        Args:
            records: List of raw records
            dataset_type: Type of dataset

        Returns:
            List of aligned records (records that couldn't be aligned are skipped)
        """
        aligned = []
        for raw in records:
            try:
                result = self.align_record(raw, dataset_type)
                if result is not None:
                    aligned.append(result)
            except Exception as e:
                # Log error but continue with other records
                print(f"Warning: Failed to align record {raw.asset_id}: {e}")
                continue
        return aligned

    def _align_native(
        self,
        ts_utc: datetime,
    ) -> tuple[datetime, date] | None:
        """
        Align native market close timestamp.

        For market_prices and volatility_indices, the raw timestamp
        is already at market close. We just need to:
        1. Verify it's a trading day
        2. Set canonical close time

        Returns:
            (aligned_timestamp, trading_date) or None if not trading day
        """
        trading_date = ts_utc.date()

        if not self._calendar.is_trading_day(trading_date):
            # Skip non-trading days
            return None

        # Get canonical close time for this date
        close_utc = self._calendar.get_market_close_utc(trading_date)

        return (close_utc, trading_date)

    def _align_to_previous_close(
        self,
        ts_utc: datetime,
    ) -> tuple[datetime, date] | None:
        """
        Align to previous trading day close.

        For macro_rates: data published on day T is aligned to T-1 close.
        This prevents lookahead bias.

        From time_alignment.yaml:
        if publication_time <= market_close(T):
            assign_to = T
        else:
            assign_to = next_trading_day(T)

        But for macro_rates with 1d lag, we adjust to T-1.
        """
        publication_date = ts_utc.date()

        # Get previous trading day
        prev_trading_day = self._calendar.get_previous_trading_day(publication_date)

        # Get close time for that day
        close_utc = self._calendar.get_market_close_utc(prev_trading_day)

        return (close_utc, prev_trading_day)

    def _align_to_same_day_close(
        self,
        ts_utc: datetime,
    ) -> tuple[datetime, date] | None:
        """
        Align to same day or next day close based on publication time.

        From time_alignment.yaml:
        - If publication_time <= market_close(T): assign_to = T
        - Else: assign_to = next_trading_day(T)

        For financial_news and sentiment.
        """
        publication_date = ts_utc.date()

        # Check if publication date is a trading day
        if self._calendar.is_trading_day(publication_date):
            close_utc = self._calendar.get_market_close_utc(publication_date)

            # Compare publication time to market close
            if ts_utc <= close_utc:
                # Published before or at close -> assign to this day
                return (close_utc, publication_date)
            else:
                # Published after close -> assign to next trading day
                next_day = self._calendar.get_next_trading_day(publication_date)
                next_close = self._calendar.get_market_close_utc(next_day)
                return (next_close, next_day)
        else:
            # Not a trading day -> assign to next trading day
            next_day = self._calendar.get_next_trading_day(publication_date)
            next_close = self._calendar.get_market_close_utc(next_day)
            return (next_close, next_day)


def create_aligner(use_fallback_calendar: bool = False) -> TimeAligner:
    """
    Factory function to create a TimeAligner.

    Args:
        use_fallback_calendar: Use simple calendar without pandas_market_calendars

    Returns:
        Configured TimeAligner instance
    """
    calendar = NYSECalendar(use_fallback=use_fallback_calendar)
    return TimeAligner(calendar=calendar)
