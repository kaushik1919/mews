"""
MEWS-FIN Data Ingestion Runner.

Orchestrates the ingestion pipeline:
1. Fetch raw data via adapters
2. Align timestamps via alignment engine
3. Validate against schema
4. Write to versioned Parquet output

Usage:
    python run_ingestion.py --dataset market_prices --dry-run
    python run_ingestion.py --dataset market_prices --tickers AAPL,MSFT,GOOGL
    python run_ingestion.py --dataset market_prices --mock  # For testing
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Add current directory to path for local imports
# This allows running from data-ingestion/ directory without pip install
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Now import using relative package structure
from adapters.market_prices import MarketPricesAdapter
from adapters.volatility_indices import VolatilityIndicesAdapter
from alignment import AlignedRecord, NYSECalendar, TimeAligner
from alignment.lag_rules import DatasetType
from outputs import records_to_dataframe, write_parquet
from schemas import SchemaValidator

UTC = ZoneInfo("UTC")


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator.

    Coordinates adapters, alignment, validation, and output.
    Designed to be extensible for future dataset types.
    """

    def __init__(
        self,
        dataset_name: str,
        use_mock: bool = False,
        use_fallback_calendar: bool = False,
    ):
        """
        Initialize pipeline for a specific dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'market_prices')
            use_mock: Use mock data instead of live API
            use_fallback_calendar: Use simple calendar (for testing)
        """
        self._dataset_name = dataset_name
        self._use_mock = use_mock

        # Initialize components
        self._calendar = NYSECalendar(use_fallback=use_fallback_calendar)
        self._aligner = TimeAligner(calendar=self._calendar)
        self._validator = SchemaValidator(dataset_name)

        # Get adapter for dataset type
        self._adapter = self._get_adapter(dataset_name)
        self._dataset_type = self._get_dataset_type(dataset_name)

    def _get_adapter(self, dataset_name: str):
        """Get the appropriate adapter for the dataset."""
        adapters = {
            "market_prices": MarketPricesAdapter(use_mock=self._use_mock),
            "volatility_indices": VolatilityIndicesAdapter(use_mock=self._use_mock),
            # Future: Add other adapters here
            # "macro_rates": MacroRatesAdapter(...),
        }

        if dataset_name not in adapters:
            raise ValueError(
                f"No adapter for dataset: {dataset_name}. "
                f"Available: {list(adapters.keys())}"
            )

        return adapters[dataset_name]

    def _get_dataset_type(self, dataset_name: str) -> DatasetType:
        """Map dataset name to DatasetType enum."""
        mapping = {
            "market_prices": DatasetType.MARKET_PRICES,
            "volatility_indices": DatasetType.VOLATILITY_INDICES,
            "macro_rates": DatasetType.MACRO_RATES,
            "financial_news": DatasetType.FINANCIAL_NEWS,
        }

        if dataset_name not in mapping:
            raise ValueError(f"Unknown dataset type: {dataset_name}")

        return mapping[dataset_name]

    def run(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
        dry_run: bool = False,
    ) -> dict:
        """
        Run the full ingestion pipeline.

        Args:
            tickers: List of ticker symbols to ingest
            start_date: Start of date range
            end_date: End of date range
            dry_run: If True, validate but don't write output

        Returns:
            Dictionary with run statistics
        """
        stats = {
            "dataset": self._dataset_name,
            "tickers": tickers,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "dry_run": dry_run,
            "raw_records": 0,
            "aligned_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
            "output_path": None,
        }

        print(f"\n{'='*60}")
        print(f"MEWS-FIN Ingestion: {self._dataset_name}")
        print(f"{'='*60}")
        print(f"Tickers: {tickers}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"Source: {'MOCK DATA' if self._use_mock else 'LIVE API'}")
        print()

        # Step 1: Fetch raw data
        print("Step 1: Fetching raw data...")
        raw_records = self._adapter.fetch(tickers, start_date, end_date)
        stats["raw_records"] = len(raw_records)
        print(f"  → Fetched {len(raw_records)} raw records")

        if len(raw_records) == 0:
            print("  ⚠ No raw records fetched. Exiting.")
            return stats

        # Step 2: Align timestamps
        print("\nStep 2: Aligning timestamps to UTC/NYSE close...")
        aligned_records = self._aligner.align_records(raw_records, self._dataset_type)
        stats["aligned_records"] = len(aligned_records)
        print(f"  → Aligned {len(aligned_records)} records")

        # Step 3: Validate against schema
        print("\nStep 3: Validating against schema...")
        valid_records, invalid_with_results = self._validator.validate_batch(aligned_records)
        stats["valid_records"] = len(valid_records)
        stats["invalid_records"] = len(invalid_with_results)

        print(f"  → Valid: {len(valid_records)}")
        print(f"  → Invalid: {len(invalid_with_results)}")

        # Report validation errors
        for record, result in invalid_with_results[:5]:  # Show first 5
            for error in result.errors:
                error_msg = f"{record.asset_id}: {error}"
                stats["errors"].append(error_msg)
                print(f"  ✗ {error_msg}")

        if len(invalid_with_results) > 5:
            print(f"  ... and {len(invalid_with_results) - 5} more errors")

        # Collect warnings
        for record in valid_records:
            result = self._validator.validate(record)
            for warning in result.warnings:
                stats["warnings"].append(f"{record.asset_id}: {warning}")

        # Step 4: Write output (unless dry run)
        if dry_run:
            print("\nStep 4: Dry run - skipping output write")
            print("  → Would write Parquet file with schema:")
            self._print_schema_summary(valid_records)
        else:
            print("\nStep 4: Writing Parquet output...")
            if len(valid_records) > 0:
                output_path = write_parquet(valid_records, self._dataset_name)
                stats["output_path"] = str(output_path)
                print(f"  → Written to: {output_path}")
            else:
                print("  ⚠ No valid records to write")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Raw records fetched: {stats['raw_records']}")
        print(f"Records aligned: {stats['aligned_records']}")
        print(f"Records valid: {stats['valid_records']}")
        print(f"Records invalid: {stats['invalid_records']}")
        if stats["output_path"]:
            print(f"Output: {stats['output_path']}")
        print()

        return stats

    def _print_schema_summary(self, records: list[AlignedRecord]) -> None:
        """Print a summary of the output schema."""
        if not records:
            print("  (no records)")
            return

        # Get DataFrame to show structure
        df = records_to_dataframe(records, dataset_name=self._dataset_name)
        print(f"  Columns: {list(df.columns)}")
        print("  Dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        print(f"  Row count: {len(df)}")

        # Show sample aligned timestamps
        if "timestamp" in df.columns and len(df) > 0:
            print("\n  Sample aligned timestamps (UTC):")
            for ts in df["timestamp"].head(3):
                print(f"    {ts}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MEWS-FIN Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with mock data
  python run_ingestion.py --dataset market_prices --mock --dry-run

  # Ingest specific tickers
  python run_ingestion.py --dataset market_prices --tickers AAPL,MSFT,GOOGL

  # Ingest last 30 days
  python run_ingestion.py --dataset market_prices --days 30
        """,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["market_prices", "volatility_indices"],
        help="Dataset to ingest",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=None,  # Default depends on dataset
        help="Comma-separated list of tickers/indices (default: dataset-specific)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Overrides --days.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today.",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of live API",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, don't write output",
    )

    parser.add_argument(
        "--fallback-calendar",
        action="store_true",
        help="Use simple calendar (no pandas_market_calendars)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse tickers with dataset-specific defaults
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        # Dataset-specific defaults (configuration-driven)
        default_tickers = {
            "market_prices": ["SPY", "QQQ", "IWM"],
            "volatility_indices": ["^VIX", "^VIX3M", "^VVIX"],
        }
        tickers = default_tickers.get(args.dataset, ["SPY"])

    # Parse dates
    end_date = datetime.now(UTC)
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        start_date = end_date - timedelta(days=args.days)

    # Create and run pipeline
    pipeline = IngestionPipeline(
        dataset_name=args.dataset,
        use_mock=args.mock,
        use_fallback_calendar=args.fallback_calendar,
    )

    stats = pipeline.run(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        dry_run=args.dry_run,
    )

    # Exit with error if there were invalid records
    if stats["invalid_records"] > 0:
        print("⚠ Some records failed validation")
        sys.exit(1)

    print("✓ Ingestion complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
