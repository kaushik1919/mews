"""
Macro rates fetch module.

Fetches raw macroeconomic series data from FRED (Federal Reserve Economic Data).
Does NOT perform time alignment - that is the aligner's job.
Does NOT perform forward-fill - that happens after alignment.

Assumptions:
- FRED publishes data with a lag (typically T+1)
- We preserve raw observation dates for alignment layer
- Adapter emits one row per (series_id, observation_date)
"""

from datetime import datetime
from typing import Any

from .. import BaseAdapter, RawRecord


class MacroRatesAdapter(BaseAdapter):
    """
    Adapter for fetching macroeconomic rates from FRED.

    This adapter:
    - Fetches daily observations for specified FRED series
    - Preserves raw observation dates (NOT publication dates)
    - Does NOT align to UTC (alignment layer handles this)
    - Does NOT forward-fill missing values
    - Does NOT compute derived metrics (spreads, curves)

    Configuration-driven: series IDs are passed at runtime, not hardcoded.

    From core-specs/datasets.yaml:
    - series_id: FRED series identifier
    - value: Rate or spread value (nullable for holidays)
    """

    # Default series if none specified (configuration-driven)
    # These are examples from datasets.yaml, not hardcoded requirements
    DEFAULT_SERIES = [
        "DGS10",          # 10-Year Treasury
        "DGS2",           # 2-Year Treasury
        "DFF",            # Fed Funds Rate
        "BAMLH0A0HYM2",   # High Yield OAS
    ]

    def __init__(self, use_mock: bool = False):
        """
        Initialize the adapter.

        Args:
            use_mock: If True, use mock data instead of live FRED API.
                      Useful for testing and CI/CD.
        """
        self._use_mock = use_mock

    @property
    def dataset_name(self) -> str:
        return "macro_rates"

    @property
    def source_name(self) -> str:
        return "fred"

    def fetch(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch raw observation data for FRED series.

        Args:
            tickers: List of FRED series IDs (e.g., ['DGS10', 'DGS2', 'DFF'])
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of RawRecord with raw observation timestamps (not aligned)
            One record per (series_id, observation_date)
        """
        if self._use_mock:
            return self._fetch_mock(tickers, start_date, end_date)
        return self._fetch_live(tickers, start_date, end_date)

    def _fetch_live(
        self,
        series_ids: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch live data from FRED API.

        Uses pandas_datareader with FRED backend.
        Requires FRED API key (free registration at fred.stlouisfed.org).
        """
        try:
            import pandas_datareader.data as web
        except ImportError as e:
            raise ImportError(
                "pandas_datareader is required for live FRED data fetching. "
                "Install with: pip install pandas-datareader"
            ) from e

        from zoneinfo import ZoneInfo

        # FRED data is published in US Eastern time
        eastern = ZoneInfo("America/New_York")

        records: list[RawRecord] = []

        for series_id in series_ids:
            try:
                # Fetch from FRED
                df = web.DataReader(
                    series_id,
                    "fred",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

                for obs_date, row in df.iterrows():
                    value = row[series_id]

                    # Skip missing values at fetch time
                    # (alignment layer will handle forward-fill)
                    if value is None or (hasattr(value, "__float__") and str(value) == "nan"):
                        import math
                        if isinstance(value, float) and math.isnan(value):
                            continue

                    # Raw timestamp: observation date at 12:00 ET
                    # (midday, before market close)
                    raw_ts = datetime(
                        obs_date.year,
                        obs_date.month,
                        obs_date.day,
                        12, 0, 0,
                        tzinfo=eastern,
                    )

                    record = RawRecord(
                        raw_timestamp=raw_ts,
                        asset_id=series_id,  # series_id maps to asset_id internally
                        data={
                            "value": float(value),
                        },
                        source="fred",
                    )
                    records.append(record)

            except Exception as e:
                print(f"Warning: Failed to fetch {series_id}: {e}")
                continue

        return records

    def _fetch_mock(
        self,
        series_ids: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Generate mock FRED data for testing.

        Creates realistic-looking rate data:
        - Treasury yields in typical ranges
        - Credit spreads in typical ranges
        - Gaps on weekends (FRED doesn't publish on weekends)
        - Some missing values to test forward-fill
        """
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        eastern = ZoneInfo("America/New_York")

        # Mock base values and ranges by series
        mock_params: dict[str, dict[str, Any]] = {
            "DGS10": {"base": 4.25, "volatility": 0.05},  # 10Y Treasury ~4.25%
            "DGS2": {"base": 4.65, "volatility": 0.05},   # 2Y Treasury ~4.65%
            "DTB3": {"base": 5.25, "volatility": 0.03},   # 3M T-Bill ~5.25%
            "DFF": {"base": 5.33, "volatility": 0.00},    # Fed Funds (target)
            "SOFR": {"base": 5.31, "volatility": 0.02},   # SOFR ~5.31%
            "BAMLH0A0HYM2": {"base": 3.50, "volatility": 0.10},  # HY OAS ~350bp
            "BAMLC0A0CM": {"base": 1.10, "volatility": 0.05},    # IG OAS ~110bp
            "TEDRATE": {"base": 0.15, "volatility": 0.02},       # TED spread
            "USD3MTD156N": {"base": 5.58, "volatility": 0.02},   # 3M LIBOR
        }

        records: list[RawRecord] = []
        current = start_date
        day_counter = 0

        while current <= end_date:
            # Skip weekends (FRED doesn't publish)
            weekday = current.weekday()
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                current += timedelta(days=1)
                continue

            for series_id in series_ids:
                params = mock_params.get(series_id, {"base": 1.0, "volatility": 0.01})

                # Simulate occasional missing data (gap every 7th day for testing)
                # This tests forward-fill logic
                if day_counter % 7 == 0 and series_id == "TEDRATE":
                    # Skip this observation - simulates gap
                    continue

                # Generate value with slight variation
                import math
                variation = math.sin(day_counter * 0.1) * params["volatility"]
                value = params["base"] + variation

                # Raw timestamp: observation date at 12:00 ET
                raw_ts = datetime(
                    current.year,
                    current.month,
                    current.day,
                    12, 0, 0,
                    tzinfo=eastern,
                )

                record = RawRecord(
                    raw_timestamp=raw_ts,
                    asset_id=series_id,
                    data={
                        "value": round(value, 4),
                    },
                    source="fred",
                )
                records.append(record)

            current += timedelta(days=1)
            day_counter += 1

        return records
