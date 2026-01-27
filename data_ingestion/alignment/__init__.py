"""
Time alignment engine.

This module implements the time alignment rules from core-specs/time_alignment.yaml.
ALL time alignment logic lives here - adapters must NOT perform alignment.

Key responsibilities:
- UTC conversion (all timestamps stored as UTC)
- NYSE market calendar handling
- Market close alignment (21:00 UTC)
- Publication lag enforcement
- Lookahead prevention
- Forward-fill for macro data (bounded, post-alignment)
"""

from .aligner import AlignedRecord, TimeAligner
from .calendar import NYSECalendar
from .forward_fill import (
    DEFAULT_FORWARD_FILL_CONFIG,
    ForwardFillConfig,
    forward_fill_series,
    generate_missing_dates,
)
from .lag_rules import LagRule, get_publication_lag

__all__ = [
    "NYSECalendar",
    "get_publication_lag",
    "LagRule",
    "TimeAligner",
    "AlignedRecord",
    "ForwardFillConfig",
    "DEFAULT_FORWARD_FILL_CONFIG",
    "forward_fill_series",
    "generate_missing_dates",
]
