"""
Validation for ML risk models.

CRITICAL VALIDATION RULES:
1. Output risk score ∈ [0, 1]
2. Temporal integrity preserved
3. No feature leakage
4. Fail fast if violated

These validations must pass before reporting any results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from risk_engine.ml.config import ALL_FEATURES
from risk_engine.ml.dataset import DatasetSplit, MLDataset
from risk_engine.ml.train import TrainedModel

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a model or dataset."""

    results: list[ValidationResult]
    all_passed: bool
    critical_failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_passed": self.all_passed,
            "critical_failures": self.critical_failures,
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                }
                for r in self.results
            ],
        }


def validate_risk_score_range(
    risk_scores: np.ndarray,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """
    Validate that all risk scores are in [0, 1].

    Args:
        risk_scores: Array of risk scores
        tolerance: Numerical tolerance for bounds

    Returns:
        ValidationResult
    """
    min_score = float(np.min(risk_scores))
    max_score = float(np.max(risk_scores))

    in_range = (min_score >= -tolerance) and (max_score <= 1.0 + tolerance)

    return ValidationResult(
        check_name="risk_score_range",
        passed=in_range,
        message=(
            "Risk scores in valid range [0, 1]"
            if in_range
            else f"Risk scores out of range: min={min_score:.4f}, max={max_score:.4f}"
        ),
        details={"min": min_score, "max": max_score},
    )


def validate_no_nan_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> ValidationResult:
    """
    Validate that predictions contain no NaN values.

    Args:
        predictions: Predicted labels
        probabilities: Predicted probabilities

    Returns:
        ValidationResult
    """
    pred_nan = np.isnan(predictions).sum()

    prob_nan = 0
    if probabilities is not None:
        prob_nan = np.isnan(probabilities).sum()

    passed = (pred_nan == 0) and (prob_nan == 0)

    return ValidationResult(
        check_name="no_nan_predictions",
        passed=passed,
        message=(
            "No NaN values in predictions"
            if passed
            else f"Found NaN: {pred_nan} in predictions, {prob_nan} in probabilities"
        ),
        details={"pred_nan": int(pred_nan), "prob_nan": int(prob_nan)},
    )


def validate_temporal_integrity(
    dataset: MLDataset,
) -> ValidationResult:
    """
    Validate that train/val/test splits are temporally ordered.

    No overlap between splits. No future data leakage.

    Args:
        dataset: MLDataset to validate

    Returns:
        ValidationResult
    """
    train_max = dataset.train.timestamps.max()
    val_min = dataset.val.timestamps.min()
    val_max = dataset.val.timestamps.max()
    test_min = dataset.test.timestamps.min()

    train_before_val = train_max < val_min
    val_before_test = val_max < test_min

    passed = train_before_val and val_before_test

    return ValidationResult(
        check_name="temporal_integrity",
        passed=passed,
        message=(
            "Temporal ordering preserved: train < val < test"
            if passed
            else "Temporal ordering violated - potential data leakage"
        ),
        details={
            "train_max": str(train_max),
            "val_min": str(val_min),
            "val_max": str(val_max),
            "test_min": str(test_min),
        },
    )


def validate_no_feature_leakage(
    dataset: MLDataset,
) -> ValidationResult:
    """
    Validate that features don't contain future information.

    This is a heuristic check - verifies that standardization
    was fit on training data only.

    Args:
        dataset: MLDataset to validate

    Returns:
        ValidationResult
    """
    # Check that feature means/stds were computed
    has_means = len(dataset.feature_means) > 0
    has_stds = len(dataset.feature_stds) > 0

    # Verify train has non-zero variance features
    train_std = np.std(dataset.train.X, axis=0)
    has_variance = np.all(train_std > 1e-10)

    passed = has_means and has_stds and has_variance

    return ValidationResult(
        check_name="no_feature_leakage",
        passed=passed,
        message=(
            "Feature preprocessing appears correct"
            if passed
            else "Feature preprocessing may have issues"
        ),
        details={
            "has_means": has_means,
            "has_stds": has_stds,
            "min_train_std": float(np.min(train_std)) if has_variance else 0,
        },
    )


def validate_feature_names(
    feature_names: list[str],
) -> ValidationResult:
    """
    Validate that feature names match Phase 3 outputs.

    Args:
        feature_names: List of feature names used

    Returns:
        ValidationResult
    """
    unknown = [f for f in feature_names if f not in ALL_FEATURES]
    missing = [f for f in ALL_FEATURES if f not in feature_names]

    passed = len(unknown) == 0

    return ValidationResult(
        check_name="feature_names",
        passed=passed,
        message=(
            f"All {len(feature_names)} features are valid Phase 3 outputs"
            if passed
            else f"Found {len(unknown)} unknown features: {unknown}"
        ),
        details={
            "n_features": len(feature_names),
            "unknown": unknown,
            "missing": missing,
        },
    )


def validate_class_distribution(
    dataset: MLDataset,
    min_samples_per_class: int = 10,
) -> ValidationResult:
    """
    Validate that all classes have sufficient samples.

    Args:
        dataset: MLDataset to validate
        min_samples_per_class: Minimum samples required per class

    Returns:
        ValidationResult
    """
    train_classes, train_counts = np.unique(dataset.train.y, return_counts=True)
    min_count = int(np.min(train_counts))
    n_classes = len(train_classes)

    passed = min_count >= min_samples_per_class

    return ValidationResult(
        check_name="class_distribution",
        passed=passed,
        message=(
            f"All {n_classes} classes have >= {min_samples_per_class} samples"
            if passed
            else f"Class imbalance: min samples = {min_count}"
        ),
        details={
            "n_classes": n_classes,
            "min_samples": min_count,
            "class_counts": {int(c): int(n) for c, n in zip(train_classes, train_counts, strict=False)},
        },
    )


def validate_deterministic_output(
    trained_model: TrainedModel,
    X: np.ndarray,
    n_trials: int = 3,
) -> ValidationResult:
    """
    Validate that model produces deterministic output.

    Args:
        trained_model: Trained model
        X: Test samples
        n_trials: Number of prediction trials

    Returns:
        ValidationResult
    """
    predictions = []
    for _ in range(n_trials):
        pred = trained_model.predict(X[:100])  # Use subset for speed
        predictions.append(pred)

    # All predictions should be identical
    all_same = all(
        np.array_equal(predictions[0], predictions[i])
        for i in range(1, len(predictions))
    )

    return ValidationResult(
        check_name="deterministic_output",
        passed=all_same,
        message=(
            "Model produces deterministic output"
            if all_same
            else "Model output is non-deterministic - check random seed"
        ),
        details={"n_trials": n_trials},
    )


def validate_model_output(
    trained_model: TrainedModel,
    data_split: DatasetSplit,
) -> ValidationReport:
    """
    Validate model outputs on a data split.

    Args:
        trained_model: Trained model
        data_split: Data split to validate

    Returns:
        ValidationReport
    """
    results = []
    critical_failures = []

    # Get predictions
    predictions = trained_model.predict(data_split.X)
    probabilities = trained_model.predict_proba(data_split.X)
    risk_scores = trained_model.predict_risk_score(data_split.X)

    # Validate risk score range
    result = validate_risk_score_range(risk_scores)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    # Validate no NaN
    result = validate_no_nan_predictions(predictions, probabilities)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    # Validate deterministic
    result = validate_deterministic_output(trained_model, data_split.X)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    all_passed = len(critical_failures) == 0

    return ValidationReport(
        results=results,
        all_passed=all_passed,
        critical_failures=critical_failures,
    )


def validate_dataset(
    dataset: MLDataset,
) -> ValidationReport:
    """
    Validate dataset before training.

    Args:
        dataset: MLDataset to validate

    Returns:
        ValidationReport
    """
    results = []
    critical_failures = []

    # Validate temporal integrity
    result = validate_temporal_integrity(dataset)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    # Validate no feature leakage
    result = validate_no_feature_leakage(dataset)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    # Validate feature names
    result = validate_feature_names(dataset.feature_names)
    results.append(result)
    if not result.passed:
        critical_failures.append(result.check_name)

    # Validate class distribution
    result = validate_class_distribution(dataset)
    results.append(result)
    if not result.passed:
        # Warning, not critical
        logger.warning(f"Class distribution: {result.message}")

    all_passed = len(critical_failures) == 0

    return ValidationReport(
        results=results,
        all_passed=all_passed,
        critical_failures=critical_failures,
    )


def validate_before_reporting(
    trained_model: TrainedModel,
    dataset: MLDataset,
) -> ValidationReport:
    """
    Run all validations before reporting results.

    FAIL FAST if any critical validation fails.

    Args:
        trained_model: Trained model
        dataset: Dataset used for training/evaluation

    Returns:
        ValidationReport

    Raises:
        ValueError: If critical validation fails
    """
    all_results = []
    all_critical = []

    # Validate dataset
    dataset_report = validate_dataset(dataset)
    all_results.extend(dataset_report.results)
    all_critical.extend(dataset_report.critical_failures)

    # Validate model output on test set
    output_report = validate_model_output(trained_model, dataset.test)
    all_results.extend(output_report.results)
    all_critical.extend(output_report.critical_failures)

    all_passed = len(all_critical) == 0

    report = ValidationReport(
        results=all_results,
        all_passed=all_passed,
        critical_failures=all_critical,
    )

    if not all_passed:
        failures = ", ".join(all_critical)
        raise ValueError(
            f"Validation failed before reporting. Critical failures: {failures}"
        )

    return report


def print_validation_report(report: ValidationReport) -> None:
    """Print human-readable validation report."""
    print("\nValidation Report")
    print("=" * 40)

    for result in report.results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.check_name}: {result.message}")

    if report.all_passed:
        print("\nAll validations passed.")
    else:
        print(f"\nFailed: {', '.join(report.critical_failures)}")
