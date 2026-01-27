"""
Market prices adapter.

Fetches daily OHLCV data from free public sources.
Primary: Yahoo Finance
Backup: Stooq (conceptual, not implemented in Phase 2.0)

This adapter emits RAW data only. Time alignment is handled
by the alignment engine, not here.
"""

from .fetch import MarketPricesAdapter
from .normalize import normalize_yahoo_response

__all__ = ["MarketPricesAdapter", "normalize_yahoo_response"]
