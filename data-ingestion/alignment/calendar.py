"""
NYSE market calendar.

Provides trading day logic for NYSE market.
Handles holidays, weekends, and early close days.

Reference: core-specs/time_alignment.yaml
- primary_market: NYSE
- close: 21:00:00Z (4:00 PM ET)
- holidays: excluded from trading day sequences
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# NYSE close time in UTC (4:00 PM ET = 21:00 UTC during EST)
# Note: During EDT (summer), close is 20:00 UTC
# We use the calendar to get the actual close time
NYSE_CLOSE_ET = time(16, 0, 0)  # 4:00 PM Eastern

# Timezone
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class NYSECalendar:
    """
    NYSE market calendar for trading day and close time logic.

    Uses pandas_market_calendars if available, falls back to
    basic heuristics for testing.

    From time_alignment.yaml:
    - close: 21:00:00Z (4:00 PM ET in UTC)
    - holidays: excluded
    - early_close: included, align to actual close
    """

    def __init__(self, use_fallback: bool = False):
        """
        Initialize calendar.

        Args:
            use_fallback: If True, use simple weekend-only logic
                         (for testing without pandas_market_calendars)
        """
        self._use_fallback = use_fallback
        self._calendar = None

        if not use_fallback:
            try:
                import pandas_market_calendars as mcal
                self._calendar = mcal.get_calendar("NYSE")
            except ImportError:
                self._use_fallback = True

    def is_trading_day(self, dt: date) -> bool:
        """
        Check if a given date is a NYSE trading day.

        Args:
            dt: Date to check

        Returns:
            True if NYSE is open on this date
        """
        if self._use_fallback:
            # Simple heuristic: weekdays only
            return dt.weekday() < 5

        # Use pandas_market_calendars
        schedule = self._calendar.schedule(
            start_date=dt.isoformat(),
            end_date=dt.isoformat(),
        )
        return len(schedule) > 0

    def get_market_close_utc(self, dt: date) -> datetime:
        """
        Get the market close time in UTC for a given trading date.

        Args:
            dt: Trading date

        Returns:
            Market close datetime in UTC

        From time_alignment.yaml:
        - Standard close: 21:00:00Z (during EST) or 20:00:00Z (during EDT)
        - Early close days use actual close time
        """
        if self._use_fallback:
            # Assume standard close at 4:00 PM ET
            eastern_close = datetime.combine(dt, NYSE_CLOSE_ET, tzinfo=EASTERN)
            return eastern_close.astimezone(UTC)

        # Use pandas_market_calendars for accurate close time
        schedule = self._calendar.schedule(
            start_date=dt.isoformat(),
            end_date=dt.isoformat(),
        )

        if len(schedule) == 0:
            raise ValueError(f"{dt} is not a trading day")

        # market_close is timezone-aware
        close_time = schedule.iloc[0]["market_close"]
        return close_time.to_pydatetime().astimezone(UTC)

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """
        Get list of trading days in a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of trading dates
        """
        if self._use_fallback:
            days = []
            current = start_date
            while current <= end_date:
                if self.is_trading_day(current):
                    days.append(current)
                current += timedelta(days=1)
            return days

        schedule = self._calendar.schedule(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        return [d.date() for d in schedule.index]

    def get_previous_trading_day(self, dt: date) -> date:
        """
        Get the trading day before the given date.

        Args:
            dt: Reference date

        Returns:
            Previous trading day
        """
        current = dt - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
            # Safety limit to prevent infinite loop
            if (dt - current).days > 10:
                raise ValueError(f"Could not find trading day before {dt}")
        return current

    def get_next_trading_day(self, dt: date) -> date:
        """
        Get the next trading day after the given date.

        Args:
            dt: Reference date

        Returns:
            Next trading day
        """
        current = dt + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
            # Safety limit
            if (current - dt).days > 10:
                raise ValueError(f"Could not find trading day after {dt}")
        return current
