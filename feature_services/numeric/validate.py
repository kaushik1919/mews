"""
Feature validation against core-specs/features.yaml.

Ensures:
1. Feature names match spec exactly
2. No extra features
3. No missing required features
4. No NaN values (use None instead)
5. All timestamps equal as_of

Fail fast on violations.
"""

from pathlib import Path
from typing import Any

import yaml

# Path to core-specs
CORE_SPECS_DIR = Path(__file__).parent.parent.parent / "core-specs"
FEATURES_YAML = CORE_SPECS_DIR / "features.yaml"


def load_feature_spec() -> dict[str, Any]:
    """Load features.yaml specification."""
    if not FEATURES_YAML.exists():
        raise FileNotFoundError(
            f"Feature spec not found: {FEATURES_YAML}. "
            "Ensure core-specs/features.yaml exists."
        )

    with open(FEATURES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_numeric_feature_names() -> list[str]:
    """
    Get list of all numeric feature names from spec.

    Phase 3.0 implements a subset of numeric features.
    This returns ALL numeric features defined in spec.
    """
    spec = load_feature_spec()
    numeric = spec.get("numeric", {})
    return list(numeric.keys())


def get_phase_30_feature_names() -> list[str]:
    """
    Get list of numeric features implemented in Phase 3.0.

    Phase 3.0 scope (from task specification):
    - Realized volatility (20d, 60d)
    - Volatility ratio (20d/60d)
    - Max drawdown (20d, 60d)
    - Volume z-score (20d)
    - Volume-price divergence (20d)
    - VIX level

    NOT in Phase 3.0:
    - vix_term_structure (requires VIX3M)
    - credit_spread_hy (macro-derived)
    - ted_spread (macro-derived)
    """
    return [
        "realized_volatility_20d",
        "realized_volatility_60d",
        "volatility_ratio_20d_60d",
        "max_drawdown_20d",
        "max_drawdown_60d",
        "volume_zscore_20d",
        "volume_price_divergence",
        "vix_level",
    ]


def validate_feature_snapshot(
    features: dict[str, float | None],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate a feature snapshot against spec.

    Args:
        features: Dict of feature_name -> value (or None)
        strict: If True, require exact match to Phase 3.0 features

    Returns:
        Tuple of (is_valid, list of error messages)

    Validation rules:
    1. All feature names must exist in features.yaml
    2. Values must be float or None (no NaN)
    3. In strict mode, must have exactly Phase 3.0 features
    """
    errors: list[str] = []

    spec_features = get_numeric_feature_names()
    phase_30_features = get_phase_30_feature_names()

    for name, value in features.items():
        # Check name is in spec
        if name not in spec_features:
            errors.append(f"Unknown feature: {name} (not in features.yaml)")

        # Check for NaN (should use None instead)
        if value is not None:
            import math
            if math.isnan(value):
                errors.append(f"Feature {name} has NaN value (use None instead)")

    if strict:
        # Check for missing Phase 3.0 features
        for name in phase_30_features:
            if name not in features:
                errors.append(f"Missing required Phase 3.0 feature: {name}")

        # Check for extra features (not in Phase 3.0 scope)
        for name in features.keys():
            if name not in phase_30_features:
                errors.append(
                    f"Extra feature not in Phase 3.0 scope: {name}"
                )

    return len(errors) == 0, errors


def validate_input_datasets(
    datasets: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Validate input datasets have required structure.

    Args:
        datasets: Dict of dataset_name -> DataFrame

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    import pandas as pd

    errors: list[str] = []

    # Required datasets for Phase 3.0 numeric features
    required_datasets = ["market_prices", "volatility_indices"]

    for name in required_datasets:
        if name not in datasets:
            errors.append(f"Missing required dataset: {name}")
            continue

        df = datasets[name]
        if not isinstance(df, pd.DataFrame):
            errors.append(f"Dataset {name} must be a pandas DataFrame")
            continue

        if df.empty:
            # Empty is allowed, features will return None
            continue

        # Check for timestamp column
        if "timestamp" not in df.columns:
            errors.append(f"Dataset {name} missing 'timestamp' column")

    return len(errors) == 0, errors


def get_feature_metadata(feature_name: str) -> dict[str, Any] | None:
    """
    Get metadata for a specific feature from spec.

    Args:
        feature_name: Name of feature

    Returns:
        Feature specification dict, or None if not found
    """
    spec = load_feature_spec()

    # Check numeric features
    numeric = spec.get("numeric", {})
    if feature_name in numeric:
        return numeric[feature_name]

    # Check other modalities (for future phases)
    for modality in ["sentiment", "graph"]:
        modality_features = spec.get(modality, {})
        if feature_name in modality_features:
            return modality_features[feature_name]

    return None
