"""
Macro rates adapter.

Fetches raw macroeconomic data from FRED.
"""

from .fetch import MacroRatesAdapter
from .normalize import normalize_fred_series

__all__ = ["MacroRatesAdapter", "normalize_fred_series"]
