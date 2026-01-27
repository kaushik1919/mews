"""
Macro rates normalization module.

Normalizes raw FRED series data into consistent format.
This module handles data quality and format normalization ONLY.
It does NOT perform time alignment or forward-fill.
"""

from typing import Any


def normalize_fred_series(
    series_id: str,
    value: float | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize a single FRED series observation.

    Args:
        series_id: FRED series identifier (e.g., 'DGS10')
        value: Observation value (rate/spread)
        metadata: Optional additional metadata

    Returns:
        Normalized data dictionary with:
        - value: Normalized numeric value (or None)

    Normalization rules:
    - Values are kept as-is (already in percentage points for rates)
    - None/NaN values remain None (forward-fill happens later)
    - No unit conversion needed (FRED provides consistent units)
    """
    import math

    # Handle NaN values
    if value is not None:
        if isinstance(value, float) and math.isnan(value):
            value = None

    return {
        "value": value,
    }


def get_series_metadata(series_id: str) -> dict[str, str]:
    """
    Get metadata about a FRED series.

    This is informational only - not used in processing.
    Actual series configuration comes from datasets.yaml.

    Args:
        series_id: FRED series identifier

    Returns:
        Dictionary with series metadata
    """
    # Known series from datasets.yaml
    series_info = {
        "DGS10": {
            "description": "10-Year Treasury Constant Maturity Rate",
            "unit": "percent",
            "frequency": "daily",
        },
        "DGS2": {
            "description": "2-Year Treasury Constant Maturity Rate",
            "unit": "percent",
            "frequency": "daily",
        },
        "DTB3": {
            "description": "3-Month Treasury Bill Secondary Market Rate",
            "unit": "percent",
            "frequency": "daily",
        },
        "DFF": {
            "description": "Effective Federal Funds Rate",
            "unit": "percent",
            "frequency": "daily",
        },
        "SOFR": {
            "description": "Secured Overnight Financing Rate",
            "unit": "percent",
            "frequency": "daily",
        },
        "BAMLH0A0HYM2": {
            "description": "ICE BofA US High Yield Option-Adjusted Spread",
            "unit": "percent",
            "frequency": "daily",
        },
        "BAMLC0A0CM": {
            "description": "ICE BofA US Corporate Investment Grade OAS",
            "unit": "percent",
            "frequency": "daily",
        },
        "TEDRATE": {
            "description": "TED Spread (LIBOR - T-Bill)",
            "unit": "percent",
            "frequency": "daily",
        },
        "USD3MTD156N": {
            "description": "3-Month LIBOR (ICE)",
            "unit": "percent",
            "frequency": "daily",
        },
    }

    return series_info.get(series_id, {
        "description": f"Unknown FRED series: {series_id}",
        "unit": "unknown",
        "frequency": "unknown",
    })
