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
"""

from .aligner import AlignedRecord, TimeAligner
from .calendar import NYSECalendar
from .lag_rules import LagRule, get_publication_lag

__all__ = [
    "NYSECalendar",
    "get_publication_lag",
    "LagRule",
    "TimeAligner",
    "AlignedRecord",
]
