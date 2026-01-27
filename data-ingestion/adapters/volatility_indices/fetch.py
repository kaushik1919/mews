"""
Volatility indices fetch module.

Fetches raw OHLC data for volatility indices (VIX and related).
Does NOT perform time alignment - that is the aligner's job.

Assumptions:
- Yahoo Finance returns VIX data with timestamps at market close (local time)
- VIX trades on CBOE but closes with NYSE (4:15 PM ET, aligned to 4:00 PM)
- We preserve the raw timestamp for alignment layer to process
"""

import sys
from datetime import datetime
from pathlib import Path

_PKG_ROOT = Path(__file__).parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from adapters import BaseAdapter, RawRecord


class VolatilityIndicesAdapter(BaseAdapter):
    """
    Adapter for fetching volatility indices from Yahoo Finance.

    This adapter:
    - Fetches daily OHLC data for volatility indices
    - Preserves raw timestamps (US Eastern, typically)
    - Does NOT align to UTC (alignment layer handles this)
    - Does NOT fill missing data

    Configuration-driven: indices are passed at runtime, not hardcoded.
    """

    # Default indices if none specified (configuration-driven)
    DEFAULT_INDICES = ["^VIX", "^VIX3M", "^VVIX"]

    def __init__(self, use_mock: bool = False):
        """
        Initialize the adapter.

        Args:
            use_mock: If True, use mock data instead of live API.
                      Useful for testing and CI/CD.
        """
        self._use_mock = use_mock

    @property
    def dataset_name(self) -> str:
        return "volatility_indices"

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    def fetch(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch raw OHLC data for volatility indices.

        Args:
            tickers: List of index symbols (e.g., ['^VIX', '^VIX3M'])
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of RawRecord with raw timestamps (not UTC-aligned)
        """
        if self._use_mock:
            return self._fetch_mock(tickers, start_date, end_date)
        return self._fetch_live(tickers, start_date, end_date)

    def _fetch_live(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch live data from Yahoo Finance.

        Uses yfinance library. Respects rate limits.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for live data fetching. "
                "Install with: pip install yfinance"
            ) from e

        records: list[RawRecord] = []

        for ticker in tickers:
            try:
                index = yf.Ticker(ticker)
                hist = index.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                )

                for idx, row in hist.iterrows():
                    # idx is a pandas Timestamp with timezone info
                    # Preserve as raw - alignment layer will handle UTC conversion
                    raw_ts = idx.to_pydatetime()

                    # Volatility indices don't have volume in the traditional sense
                    # We omit fields not in the schema (volume, adjusted_close)
                    record = RawRecord(
                        raw_timestamp=raw_ts,
                        asset_id=ticker,
                        data={
                            "open": float(row["Open"]) if not _is_nan(row["Open"]) else None,
                            "high": float(row["High"]) if not _is_nan(row["High"]) else None,
                            "low": float(row["Low"]) if not _is_nan(row["Low"]) else None,
                            "close": float(row["Close"]),
                        },
                        source=self.source_name,
                    )
                    records.append(record)

            except Exception as e:
                # Log error but continue with other indices
                print(f"Warning: Failed to fetch {ticker}: {e}")
                continue

        return records

    def _fetch_mock(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Generate mock data for testing.

        Mock data simulates Yahoo Finance response structure.
        Timestamps are in US Eastern (typical Yahoo response).
        VIX typically ranges 10-40, with occasional spikes.
        """
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        eastern = ZoneInfo("America/New_York")
        records: list[RawRecord] = []

        # Generate daily records
        current = start_date
        while current <= end_date:
            # Skip weekends (simple heuristic - not NYSE calendar)
            if current.weekday() < 5:  # Mon-Fri
                for ticker in tickers:
                    # Mock VIX values with deterministic variation
                    day_offset = (current - start_date).days

                    # Base VIX value depends on ticker
                    if ticker == "^VIX":
                        base_value = 18.0
                    elif ticker == "^VIX3M":
                        base_value = 20.0  # 3-month typically higher
                    elif ticker == "^VVIX":
                        base_value = 85.0  # VVIX (VIX of VIX) is higher scale
                    else:
                        base_value = 15.0 + (hash(ticker) % 20)

                    # Daily variation (deterministic)
                    daily_var = ((day_offset * 7) % 10) / 5.0  # ±2 points

                    close = base_value + daily_var
                    open_val = close - 0.3
                    high = close + 1.5
                    low = close - 1.0

                    # VIX close time is 4:15 PM ET but aligns to 4:00 PM NYSE
                    raw_ts = current.replace(
                        hour=16, minute=0, second=0, microsecond=0,
                        tzinfo=eastern
                    )

                    record = RawRecord(
                        raw_timestamp=raw_ts,
                        asset_id=ticker,
                        data={
                            "open": open_val,
                            "high": high,
                            "low": low,
                            "close": close,
                        },
                        source=self.source_name,
                    )
                    records.append(record)

            current += timedelta(days=1)

        return records


def _is_nan(value) -> bool:
    """Check if a value is NaN (works for various numeric types)."""
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
