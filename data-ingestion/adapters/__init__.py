"""
Data source adapters.

Each adapter is responsible for:
- Fetching raw data from a public source
- Parsing raw fields only
- Emitting raw timestamps (source timezone, not aligned)
- NO alignment logic
- NO aggregation logic

Adapters are source-specific, alignment-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RawRecord:
    """
    A single raw record from a data source.

    This is the output of adapters BEFORE alignment.
    raw_timestamp is in the source's native timezone.
    """
    raw_timestamp: datetime  # Source timezone, NOT aligned
    asset_id: str
    data: dict[str, Any]
    source: str

    def __post_init__(self):
        """Validate raw record structure."""
        if self.raw_timestamp is None:
            raise ValueError("raw_timestamp cannot be None")
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty")


class BaseAdapter(ABC):
    """
    Abstract base class for all data adapters.

    Adapters fetch raw data and emit RawRecord instances.
    They must NOT perform any time alignment.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name (e.g., 'market_prices')."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source name (e.g., 'yahoo_finance')."""
        pass

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch raw data for the given tickers and date range.

        Args:
            tickers: List of ticker symbols to fetch
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of RawRecord instances with raw (unaligned) timestamps
        """
        pass
