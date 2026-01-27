"""
Volatility indices normalization module.

Handles field normalization for volatility index data.
Ensures consistent field names and types per datasets.yaml schema.
"""

from typing import Any


def normalize_volatility_record(raw: dict[str, Any], index_id: str) -> dict[str, Any]:
    """
    Normalize a raw Yahoo Finance response into standard field names.

    Args:
        raw: Raw dictionary from yfinance
        index_id: The index symbol (e.g., '^VIX')

    Returns:
        Normalized dictionary with standard field names per datasets.yaml
    """
    # Standard field mapping for volatility indices
    # Yahoo Finance may use different casing
    field_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
    }

    normalized = {}

    for raw_field, value in raw.items():
        if raw_field in field_mapping:
            normalized_field = field_mapping[raw_field]
            # Convert to float, handling NaN
            if value is not None:
                try:
                    import math
                    float_val = float(value)
                    if not math.isnan(float_val):
                        normalized[normalized_field] = float_val
                except (TypeError, ValueError):
                    pass

    return normalized


def validate_volatility_record(data: dict[str, Any]) -> list[str]:
    """
    Validate a raw record against quality rules from datasets.yaml.

    Returns list of validation errors (empty if valid).

    Quality rules from datasets.yaml:
    - Reject if close < 0
    - VIX typically ranges 10-80, flag outliers > 100
    """
    errors = []

    close = data.get("close")

    # Reject if close < 0
    if close is not None and close < 0:
        errors.append(f"close must be >= 0, got {close}")

    return errors


def get_volatility_warnings(data: dict[str, Any]) -> list[str]:
    """
    Generate warnings for unusual but valid values.

    From datasets.yaml:
    - VIX typically ranges 10-80, flag outliers > 100
    """
    warnings = []

    close = data.get("close")

    # Flag if VIX > 100 (unusual but can happen in crisis)
    if close is not None and close > 100:
        warnings.append(f"VIX close={close} is unusually high (>100)")

    return warnings
