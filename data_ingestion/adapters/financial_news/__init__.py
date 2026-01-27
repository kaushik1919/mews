"""
Financial news adapter.

Fetches raw financial news articles from public sources.
"""

from .fetch import FinancialNewsAdapter
from .normalize import normalize_article

__all__ = ["FinancialNewsAdapter", "normalize_article"]
