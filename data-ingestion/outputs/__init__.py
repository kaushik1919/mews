"""
Dataset output storage.

Handles writing validated, aligned records to versioned Parquet files.
Output format follows core-specs/datasets.yaml:
- storage_format: parquet
- versioning: date_partitioned
"""

from pathlib import Path
from typing import Any

from alignment import AlignedRecord

# Output directory relative to data-ingestion/
OUTPUTS_DIR = Path(__file__).parent / "datasets"


def get_output_path(
    dataset_name: str,
    version: str = "v1",
    filename: str = "daily.parquet",
) -> Path:
    """
    Get the output path for a dataset.

    Args:
        dataset_name: Name of dataset (e.g., 'market_prices')
        version: Version string (e.g., 'v1')
        filename: Output filename

    Returns:
        Path to output file
    """
    return OUTPUTS_DIR / dataset_name / version / filename


def records_to_dataframe(
    records: list[AlignedRecord],
    dataset_name: str = "market_prices",
) -> Any:
    """
    Convert aligned records to a pandas DataFrame.

    Args:
        records: List of aligned records
        dataset_name: Name of dataset (for correct id field name)

    Returns:
        pandas DataFrame with flattened structure
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for DataFrame operations. "
            "Install with: pip install pandas"
        ) from e

    # Determine the correct id field name per datasets.yaml
    id_field_mapping = {
        "market_prices": "asset_id",
        "volatility_indices": "index_id",
        "macro_rates": "series_id",
        "financial_news": "article_id",
    }
    id_field_name = id_field_mapping.get(dataset_name, "asset_id")

    rows = []
    for record in records:
        row = {
            "timestamp": record.timestamp,
            id_field_name: record.asset_id,  # Map to correct field name
            "source": record.source,
            **record.data,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure correct dtypes per datasets.yaml
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("int64")

    # Sort by timestamp and id field (primary key)
    if len(df) > 0:
        df = df.sort_values(["timestamp", id_field_name]).reset_index(drop=True)

    return df


def write_parquet(
    records: list[AlignedRecord],
    dataset_name: str,
    version: str = "v1",
) -> Path:
    """
    Write aligned records to a Parquet file.

    Args:
        records: List of validated, aligned records
        dataset_name: Name of dataset
        version: Version string

    Returns:
        Path to written file
    """
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pandas is required for Parquet output. "
            "Install with: pip install pandas pyarrow"
        ) from e

    output_path = get_output_path(dataset_name, version)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = records_to_dataframe(records, dataset_name=dataset_name)

    # Write Parquet with pyarrow
    df.to_parquet(output_path, engine="pyarrow", index=False)

    return output_path


def read_parquet(
    dataset_name: str,
    version: str = "v1",
) -> Any:
    """
    Read a dataset from Parquet.

    Args:
        dataset_name: Name of dataset
        version: Version string

    Returns:
        pandas DataFrame
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required") from e

    input_path = get_output_path(dataset_name, version)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    return pd.read_parquet(input_path)
