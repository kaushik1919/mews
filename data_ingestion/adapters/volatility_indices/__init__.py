"""
Volatility indices adapter package.

Fetches raw volatility index data (VIX and related).
Does NOT perform time alignment.
"""

from .fetch import VolatilityIndicesAdapter

__all__ = ["VolatilityIndicesAdapter"]
