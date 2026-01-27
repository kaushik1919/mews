"""
Schema validation module.

Validates aligned records against schemas defined in core-specs/datasets.yaml.
Ensures all output data conforms to the constitutional contract.

Validation is STRICT:
- Missing required fields -> reject
- Type mismatches -> reject
- Quality rule violations -> reject or flag per spec
- No silent coercion
"""

from .validate import (
    SchemaValidator,
    ValidationError,
    ValidationResult,
    load_dataset_schema,
)

__all__ = [
    "SchemaValidator",
    "ValidationResult",
    "ValidationError",
    "load_dataset_schema",
]
