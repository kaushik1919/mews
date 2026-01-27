"""
Schema validation against core-specs/datasets.yaml.

This module loads dataset schemas from core-specs and validates
aligned records against them.

Validation philosophy:
- Fail hard on violations
- No silent coercion
- Log warnings for flags (per spec)
"""

# Use relative import within the package
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_PKG_ROOT = Path(__file__).parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from alignment import AlignedRecord

# Path to core-specs (relative to repo root)
CORE_SPECS_DIR = Path(__file__).parent.parent.parent / "core-specs"
DATASETS_YAML = CORE_SPECS_DIR / "datasets.yaml"


@dataclass
class ValidationError:
    """A single validation error."""
    field: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.field}: {self.message} (value: {self.value})"
        return f"{self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a record."""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


def load_dataset_schema(dataset_name: str) -> dict[str, Any]:
    """
    Load schema for a dataset from core-specs/datasets.yaml.

    Args:
        dataset_name: Name of dataset (e.g., 'market_prices')

    Returns:
        Schema dictionary from datasets.yaml

    Raises:
        FileNotFoundError: If datasets.yaml doesn't exist
        KeyError: If dataset not found in spec
    """
    if not DATASETS_YAML.exists():
        raise FileNotFoundError(
            f"Core spec not found: {DATASETS_YAML}. "
            "Ensure core-specs/datasets.yaml exists."
        )

    with open(DATASETS_YAML, encoding="utf-8") as f:
        specs = yaml.safe_load(f)

    if dataset_name not in specs:
        available = [k for k in specs.keys() if not k.startswith("_")]
        raise KeyError(
            f"Dataset '{dataset_name}' not found in datasets.yaml. "
            f"Available: {available}"
        )

    return specs[dataset_name]


class SchemaValidator:
    """
    Validates aligned records against dataset schemas.

    Uses schemas from core-specs/datasets.yaml as the source of truth.
    """

    def __init__(self, dataset_name: str):
        """
        Initialize validator for a specific dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'market_prices')
        """
        self._dataset_name = dataset_name
        self._schema = load_dataset_schema(dataset_name)
        self._field_specs = self._schema.get("schema", {}).get("fields", {})
        self._quality_rules = self._schema.get("quality_rules", [])

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def required_fields(self) -> list[str]:
        """Get list of required field names."""
        return [
            name for name, spec in self._field_specs.items()
            if spec.get("required", False)
        ]

    @property
    def field_types(self) -> dict[str, str]:
        """Get mapping of field names to expected types."""
        return {
            name: spec.get("type", "any")
            for name, spec in self._field_specs.items()
        }

    def validate(self, record: AlignedRecord) -> ValidationResult:
        """
        Validate an aligned record against the schema.

        Args:
            record: Aligned record to validate

        Returns:
            ValidationResult with is_valid, errors, and warnings
        """
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Combine record data with metadata fields
        full_data = {
            "timestamp": record.timestamp,
            "asset_id": record.asset_id,
            **record.data,
        }

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in full_data:
                errors.append(ValidationError(
                    field=field_name,
                    message="Missing required field",
                ))
            elif full_data[field_name] is None:
                spec = self._field_specs.get(field_name, {})
                if not spec.get("nullable", True):
                    errors.append(ValidationError(
                        field=field_name,
                        message="Field is not nullable but got None",
                    ))

        # Check field types
        for field_name, value in full_data.items():
            if field_name not in self._field_specs:
                # Extra field - warn but don't fail
                warnings.append(f"Unexpected field: {field_name}")
                continue

            if value is None:
                continue  # Already checked nullability above

            expected_type = self._field_specs[field_name].get("type")
            if expected_type and not self._check_type(value, expected_type):
                errors.append(ValidationError(
                    field=field_name,
                    message=f"Type mismatch: expected {expected_type}, got {type(value).__name__}",
                    value=value,
                ))

        # Apply quality rules
        quality_result = self._check_quality_rules(full_data)
        errors.extend(quality_result["errors"])
        warnings.extend(quality_result["warnings"])

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_batch(
        self,
        records: list[AlignedRecord],
    ) -> tuple[list[AlignedRecord], list[tuple[AlignedRecord, ValidationResult]]]:
        """
        Validate a batch of records.

        Args:
            records: List of aligned records

        Returns:
            Tuple of (valid_records, invalid_with_results)
        """
        valid = []
        invalid = []

        for record in records:
            result = self.validate(record)
            if result.is_valid:
                valid.append(record)
            else:
                invalid.append((record, result))

        return valid, invalid

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected type from schema.

        Args:
            value: Value to check
            expected_type: Type string from schema

        Returns:
            True if type matches
        """
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "float64": lambda v: isinstance(v, (int, float)),
            "int64": lambda v: isinstance(v, int),
            "datetime": lambda v: isinstance(v, datetime),
        }

        checker = type_checks.get(expected_type)
        if checker is None:
            # Unknown type, assume valid
            return True

        return checker(value)

    def _check_quality_rules(
        self,
        data: dict[str, Any],
    ) -> dict[str, list]:
        """
        Apply quality rules from datasets.yaml.

        For market_prices:
        - Reject if close <= 0
        - Reject if high < low
        - Reject if open or close outside [low, high]
        - Flag if volume = 0
        """
        errors = []
        warnings = []

        # Only apply rules if we have the necessary fields
        if self._dataset_name == "market_prices":
            close = data.get("close")
            high = data.get("high")
            low = data.get("low")
            open_price = data.get("open")
            volume = data.get("volume")

            # Reject if close <= 0
            if close is not None and close <= 0:
                errors.append(ValidationError(
                    field="close",
                    message="close must be > 0",
                    value=close,
                ))

            # Reject if high < low
            if high is not None and low is not None and high < low:
                errors.append(ValidationError(
                    field="high",
                    message="high must be >= low",
                    value=f"high={high}, low={low}",
                ))

            # Reject if open outside [low, high]
            if all(v is not None for v in [open_price, low, high]):
                if open_price < low or open_price > high:
                    errors.append(ValidationError(
                        field="open",
                        message="open must be within [low, high]",
                        value=f"open={open_price}, low={low}, high={high}",
                    ))

            # Reject if close outside [low, high]
            if all(v is not None for v in [close, low, high]):
                if close < low or close > high:
                    errors.append(ValidationError(
                        field="close",
                        message="close must be within [low, high]",
                        value=f"close={close}, low={low}, high={high}",
                    ))

            # Flag (warn) if volume = 0
            if volume is not None and volume == 0:
                warnings.append("volume is 0 (may be valid for some instruments)")

        return {"errors": errors, "warnings": warnings}


def validate_aligned_records(
    records: list[AlignedRecord],
    dataset_name: str,
) -> tuple[list[AlignedRecord], list[tuple[AlignedRecord, ValidationResult]]]:
    """
    Convenience function to validate a batch of aligned records.

    Args:
        records: List of aligned records
        dataset_name: Name of dataset for schema lookup

    Returns:
        Tuple of (valid_records, invalid_with_results)
    """
    validator = SchemaValidator(dataset_name)
    return validator.validate_batch(records)
