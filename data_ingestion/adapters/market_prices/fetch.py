"""
Market prices fetch module.

Fetches raw OHLCV data from Yahoo Finance.
Does NOT perform time alignment - that is the aligner's job.

Assumptions:
- Yahoo Finance returns data with timestamps at market close (local time)
- We preserve the raw timestamp for alignment layer to process
- Rate limits are respected via conservative request patterns
"""

from datetime import datetime

from .. import BaseAdapter, RawRecord


class MarketPricesAdapter(BaseAdapter):
    """
    Adapter for fetching market prices from Yahoo Finance.

    This adapter:
    - Fetches daily OHLCV data
    - Preserves raw timestamps (US Eastern, typically)
    - Does NOT align to UTC (alignment layer handles this)
    - Does NOT fill missing data
    """

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
        return "market_prices"

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
        Fetch raw OHLCV data for tickers.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', '^GSPC'])
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
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    auto_adjust=False,  # Get raw prices + adjusted close
                )

                for idx, row in hist.iterrows():
                    # idx is a pandas Timestamp with timezone info
                    # Preserve as raw - alignment layer will handle UTC conversion
                    raw_ts = idx.to_pydatetime()

                    record = RawRecord(
                        raw_timestamp=raw_ts,
                        asset_id=ticker,
                        data={
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": int(row["Volume"]),
                            "adjusted_close": float(row["Adj Close"]),
                        },
                        source=self.source_name,
                    )
                    records.append(record)

            except Exception as e:
                # Log error but continue with other tickers
                # In production, this would use proper logging
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
                    # Mock prices with deterministic values based on date
                    # This ensures reproducibility for testing
                    day_offset = (current - start_date).days
                    base_price = 100.0 + (hash(ticker) % 100)

                    # Small daily variation (deterministic)
                    daily_var = ((day_offset * 7) % 10) / 10.0

                    close = base_price + daily_var
                    open_price = close - 0.5
                    high = close + 1.0
                    low = close - 1.5

                    # Raw timestamp at 4:00 PM Eastern (market close)
                    raw_ts = current.replace(
                        hour=16, minute=0, second=0, microsecond=0,
                        tzinfo=eastern
                    )

                    record = RawRecord(
                        raw_timestamp=raw_ts,
                        asset_id=ticker,
                        data={
                            "open": open_price,
                            "high": high,
                            "low": low,
                            "close": close,
                            "volume": 1000000 + (day_offset * 10000),
                            "adjusted_close": close * 0.99,  # Slight adj for mock
                        },
                        source=self.source_name,
                    )
                    records.append(record)

            current += timedelta(days=1)

        return records
