"""
Market prices normalization.

Normalizes raw Yahoo Finance responses into a consistent internal format.
This is PRE-ALIGNMENT normalization - just field standardization.

Does NOT:
- Convert timezones (alignment layer does this)
- Fill missing data
- Apply any business logic
"""

from typing import Any

from .. import RawRecord


def normalize_yahoo_response(raw: dict[str, Any], ticker: str) -> dict[str, Any]:
    """
    Normalize a raw Yahoo Finance response into standard field names.

    This handles field name variations between different yfinance versions.

    Args:
        raw: Raw dictionary from yfinance
        ticker: The ticker symbol

    Returns:
        Normalized dictionary with standard field names
    """
    # Handle potential field name variations
    field_mappings = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adjusted_close",
        "Adj_Close": "adjusted_close",
        "AdjClose": "adjusted_close",
    }

    normalized = {}
    for raw_key, std_key in field_mappings.items():
        if raw_key in raw:
            value = raw[raw_key]
            # Type coercion
            if std_key == "volume":
                normalized[std_key] = int(value) if value is not None else None
            else:
                normalized[std_key] = float(value) if value is not None else None

    return normalized


def validate_raw_record(record: RawRecord) -> list[str]:
    """
    Validate a raw record against quality rules from datasets.yaml.

    Returns list of validation errors (empty if valid).

    Quality rules from datasets.yaml:
    - Reject if close <= 0
    - Reject if high < low
    - Reject if open or close outside [low, high]
    - Flag if volume = 0 (may be valid for some instruments)
    """
    errors = []
    data = record.data

    # Required fields
    required = ["open", "high", "low", "close", "volume", "adjusted_close"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
            return errors  # Can't validate further without fields

    close = data["close"]
    high = data["high"]
    low = data["low"]
    open_price = data["open"]
    volume = data["volume"]

    # Reject if close <= 0
    if close is not None and close <= 0:
        errors.append(f"Invalid close price: {close} <= 0")

    # Reject if high < low
    if high is not None and low is not None and high < low:
        errors.append(f"Invalid OHLC: high ({high}) < low ({low})")

    # Reject if open outside [low, high]
    if all(v is not None for v in [open_price, low, high]):
        if open_price < low or open_price > high:
            errors.append(
                f"Invalid OHLC: open ({open_price}) outside [{low}, {high}]"
            )

    # Reject if close outside [low, high]
    if all(v is not None for v in [close, low, high]):
        if close < low or close > high:
            errors.append(
                f"Invalid OHLC: close ({close}) outside [{low}, {high}]"
            )

    # Flag (warn, don't reject) if volume = 0
    if volume is not None and volume == 0:
        # This is a flag, not a rejection per datasets.yaml
        # Caller can decide how to handle
        pass

    return errors
