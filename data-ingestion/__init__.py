"""
MEWS-FIN Data Ingestion Framework.

This package implements the data ingestion layer for MEWS-FIN.
All ingestion follows the contracts defined in core-specs/.

Architecture:
- adapters/: Fetch raw data from sources (no alignment)
- alignment/: Time alignment engine (UTC, NYSE close)
- schemas/: Validation against datasets.yaml
- outputs/: Versioned dataset storage

Phase 2.0 implements market_prices only.
Future phases will add: volatility_indices, macro_rates, financial_news.
"""

__version__ = "0.1.0"
