"""
Financial news normalization module.

Normalizes raw article data into consistent format.
This module handles text cleaning and format normalization ONLY.
It does NOT perform sentiment analysis or aggregation.
"""

import re
from typing import Any


def normalize_article(
    headline: str,
    body: str | None = None,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize a single news article.

    Args:
        headline: Article headline
        body: Article body text (optional)
        url: Original article URL (optional)
        metadata: Optional additional metadata

    Returns:
        Normalized data dictionary with:
        - headline: Cleaned headline
        - body: Cleaned body (or None)
        - url: Original URL

    Normalization rules:
    - Strip leading/trailing whitespace
    - Normalize internal whitespace
    - Remove HTML tags if present
    - Preserve original text (no sentiment modification)
    """
    return {
        "headline": clean_text(headline) if headline else "",
        "body": clean_text(body) if body else None,
        "url": url,
    }


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Operations:
    - Remove HTML tags
    - Normalize whitespace
    - Strip leading/trailing spaces

    Does NOT:
    - Lowercase (preserves case for sentiment)
    - Remove punctuation (important for meaning)
    - Truncate (length limits applied elsewhere)
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace (multiple spaces, newlines, tabs -> single space)
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def validate_headline(headline: str) -> bool:
    """
    Validate that a headline meets minimum requirements.

    From datasets.yaml:
    - headline is required
    - max_length: 500

    Args:
        headline: Headline text to validate

    Returns:
        True if headline is valid
    """
    if not headline:
        return False

    if len(headline.strip()) == 0:
        return False

    if len(headline) > 500:
        return False

    return True


def validate_body(body: str | None) -> bool:
    """
    Validate body text if present.

    From datasets.yaml:
    - body is optional (nullable: true)
    - max_length: 50000

    Args:
        body: Body text to validate (can be None)

    Returns:
        True if body is valid (including None)
    """
    if body is None:
        return True  # Nullable is allowed

    if len(body) > 50000:
        return False

    return True
