"""
Financial news fetch module.

Fetches raw financial news articles from public sources.
Does NOT perform time alignment - that is the aligner's job.
Does NOT perform sentiment analysis - that happens in feature engineering.
Does NOT aggregate articles - each article is a discrete event.

Assumptions:
- News is event-based, not continuous time series
- Each article has a publication timestamp
- Alignment to market time happens in alignment layer
- Sentiment inference happens in feature layer (Phase 3)
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any

from .. import BaseAdapter, RawRecord


class FinancialNewsAdapter(BaseAdapter):
    """
    Adapter for fetching financial news articles.

    This adapter:
    - Fetches raw articles from public sources
    - Preserves raw publication timestamps
    - Does NOT align to market time (alignment layer handles this)
    - Does NOT compute sentiment (feature layer handles this)
    - Does NOT aggregate articles (each is a discrete event)

    Configuration-driven: sources are passed at runtime, not hardcoded.

    From core-specs/datasets.yaml:
    - article_id: Unique identifier
    - timestamp: Publication timestamp (UTC)
    - source: News source name
    - headline: Article headline
    - body: Article body text (optional)
    - url: Original article URL
    """

    # Default sources if none specified
    DEFAULT_SOURCES = ["common_crawl", "reuters", "cnbc"]

    def __init__(self, use_mock: bool = False):
        """
        Initialize the adapter.

        Args:
            use_mock: If True, use mock data instead of live sources.
                      Useful for testing and CI/CD.
        """
        self._use_mock = use_mock

    @property
    def dataset_name(self) -> str:
        return "financial_news"

    @property
    def source_name(self) -> str:
        return "mixed"  # Multiple sources

    def fetch(
        self,
        tickers: list[str],  # For news, these are source names or keywords
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch raw news articles.

        Args:
            tickers: List of source names or filter keywords
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of RawRecord with raw publication timestamps
            One record per article (no aggregation)
        """
        if self._use_mock:
            return self._fetch_mock(tickers, start_date, end_date)
        return self._fetch_live(tickers, start_date, end_date)

    def _fetch_live(
        self,
        sources: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Fetch live news from public sources.

        In a production implementation, this would:
        - Query Common Crawl News dataset via S3/HTTP
        - Parse RSS feeds from public sources
        - Apply domain filtering for financial content

        For now, raises NotImplementedError as live fetching
        requires infrastructure setup beyond this phase.
        """
        raise NotImplementedError(
            "Live news fetching requires Common Crawl infrastructure. "
            "Use --mock for testing. Live implementation planned for Phase 3."
        )

    def _fetch_mock(
        self,
        sources: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawRecord]:
        """
        Generate mock news articles for testing.

        Creates realistic news data patterns:
        - Multiple articles per day
        - Varying publication times (before/after market close)
        - Weekend articles (for alignment testing)
        - Mixed sources
        - Varying content lengths
        """
        from zoneinfo import ZoneInfo

        utc = ZoneInfo("UTC")

        # Sample headlines for realistic mock data
        sample_headlines = [
            "Markets rally as tech stocks surge on AI optimism",
            "Federal Reserve signals potential rate pause",
            "Bond yields spike amid inflation concerns",
            "Oil prices drop on weak demand outlook",
            "Banking sector faces pressure from credit concerns",
            "Retail sales exceed expectations, boosting sentiment",
            "Cryptocurrency volatility returns as regulations loom",
            "Housing market shows signs of cooling",
            "Earnings season kicks off with mixed results",
            "Global markets diverge on economic data",
            "Trade tensions ease, lifting market sentiment",
            "Employment data surprises to the upside",
            "Consumer confidence index falls unexpectedly",
            "Manufacturing PMI indicates expansion",
            "Energy sector leads market gains",
        ]

        # Sample body texts (abbreviated for mock)
        sample_bodies = [
            "Financial analysts noted significant movement in major indices "
            "as investors digested the latest economic data. The session saw "
            "heightened volatility across multiple sectors.",
            "Market participants are closely monitoring central bank signals "
            "as policymakers navigate the current economic landscape. "
            "Expectations for future rate decisions remain uncertain.",
            "Trading volume was elevated compared to the 20-day average, "
            "with notable activity in options markets. Institutional flows "
            "suggested positioning ahead of upcoming catalysts.",
            None,  # Some articles are headline-only (RSS feeds)
            "Sector rotation continued as investors reassessed growth "
            "expectations for the coming quarters. Defensive names outperformed "
            "amid the cautious sentiment.",
        ]

        source_names = {
            "common_crawl": "common_crawl",
            "reuters": "reuters",
            "cnbc": "cnbc",
            "ft": "financial_times",
        }

        records: list[RawRecord] = []
        current = start_date
        article_counter = 0

        while current <= end_date:
            current_date = current.date()

            # Generate multiple articles per day (3-8 articles)
            num_articles = 3 + (hash(str(current_date)) % 6)

            for i in range(num_articles):
                # Vary publication times throughout the day
                # Some before market close (21:00 UTC), some after
                base_hour = 8 + (i * 2) + (hash(str(current_date) + str(i)) % 3)
                base_minute = hash(str(current_date) + str(i) + "m") % 60

                # Clamp to reasonable hours (8:00 - 23:00 UTC)
                pub_hour = min(23, max(8, base_hour))

                pub_time = datetime(
                    current_date.year,
                    current_date.month,
                    current_date.day,
                    pub_hour,
                    base_minute,
                    0,
                    tzinfo=utc,
                )

                # Select source (rotate through requested sources)
                source_key = sources[i % len(sources)] if sources else "common_crawl"
                source_name = source_names.get(source_key, source_key)

                # Generate deterministic article ID
                article_id = self._generate_article_id(
                    source_name, pub_time, article_counter
                )

                # Select headline and body
                headline_idx = (article_counter + hash(str(current_date))) % len(
                    sample_headlines
                )
                body_idx = article_counter % len(sample_bodies)

                headline = sample_headlines[headline_idx]
                body = sample_bodies[body_idx]

                # Generate URL
                url = f"https://example.com/{source_name}/article/{article_id}"

                record = RawRecord(
                    raw_timestamp=pub_time,
                    asset_id=article_id,  # article_id maps to asset_id internally
                    data={
                        "headline": headline,
                        "body": body,
                        "url": url,
                    },
                    source=source_name,
                )
                records.append(record)
                article_counter += 1

            current += timedelta(days=1)

        return records

    def _generate_article_id(
        self,
        source: str,
        timestamp: datetime,
        counter: int,
    ) -> str:
        """
        Generate a deterministic, unique article ID.

        Format: {source}_{date}_{hash}
        This ensures reproducibility in tests.
        """
        date_str = timestamp.strftime("%Y%m%d")
        hash_input = f"{source}_{timestamp.isoformat()}_{counter}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{source}_{date_str}_{hash_suffix}"
