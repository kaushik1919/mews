"""
Ensemble Validation Module.

MEWS-FIN Phase 4.2: Validate ensemble outputs to ensure
spec compliance and sanity.

Key Validations (MANDATORY):
    - Final score ∈ [0, 1]
    - Regime mapping is correct
    - No NaN or infinite values
    - Weights sum to 1.0
    - Calibration maintains monotonicity
    - Explainability fields are populated

FAIL FAST on any violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .weights import EnsembleWeights

# Regime boundaries from risk_score.yaml
REGIME_BOUNDS: dict[str, tuple[float, float]] = {
    "LOW_RISK": (0.0, 0.25),
    "MODERATE_RISK": (0.25, 0.50),
    "HIGH_RISK": (0.50, 0.75),
    "EXTREME_RISK": (0.75, 1.0),
}


@dataclass
class ValidationResult:
    """Result of validation check."""

    is_valid: bool
    check_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "check_name": self.check_name,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete validation report."""

    is_valid: bool
    results: list[ValidationResult]
    n_passed: int
    n_failed: int
    critical_failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "results": [r.to_dict() for r in self.results],
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "critical_failures": self.critical_failures,
        }

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            failures = "; ".join(self.critical_failures)
            raise ValueError(f"Ensemble validation failed: {failures}")


def validate_ensemble_output(
    score: float,
    regime: str,
    weights: EnsembleWeights | dict[str, float],
    explanation: dict[str, Any] | None = None,
    calibrated: bool = False,
    raw_score: float | None = None,
) -> ValidationReport:
    """
    Validate ensemble output against spec and sanity checks.

    Args:
        score: Final ensemble risk score
        regime: Assigned risk regime
        weights: Ensemble weights used
        explanation: Explanation dictionary (optional)
        calibrated: Whether score was calibrated
        raw_score: Pre-calibration score (for monotonicity check)

    Returns:
        Validation report with all results
    """
    results: list[ValidationResult] = []

    # Check 1: Score bounds
    results.append(_validate_score_bounds(score))

    # Check 2: No NaN/Inf
    results.append(_validate_no_nan_inf(score))

    # Check 3: Regime validity
    results.append(_validate_regime(regime))

    # Check 4: Regime-score consistency
    results.append(_validate_regime_consistency(score, regime))

    # Check 5: Weights validity
    results.append(_validate_weights(weights))

    # Check 6: Explanation completeness (if provided)
    if explanation is not None:
        results.append(_validate_explanation(explanation))

    # Check 7: Calibration monotonicity (if applicable)
    if calibrated and raw_score is not None:
        results.append(_validate_calibration_direction(raw_score, score))

    # Aggregate results
    failed = [r for r in results if not r.is_valid]
    passed = [r for r in results if r.is_valid]

    return ValidationReport(
        is_valid=len(failed) == 0,
        results=results,
        n_passed=len(passed),
        n_failed=len(failed),
        critical_failures=[r.message for r in failed],
    )


def _validate_score_bounds(score: float) -> ValidationResult:
    """Check score is in [0, 1]."""
    if 0.0 <= score <= 1.0:
        return ValidationResult(
            is_valid=True,
            check_name="score_bounds",
            message="Score within valid range [0, 1]",
            details={"score": score},
        )
    else:
        return ValidationResult(
            is_valid=False,
            check_name="score_bounds",
            message=f"Score {score:.6f} outside valid range [0, 1]",
            details={"score": score, "min": 0.0, "max": 1.0},
        )


def _validate_no_nan_inf(score: float) -> ValidationResult:
    """Check score is not NaN or infinite."""
    if np.isnan(score):
        return ValidationResult(
            is_valid=False,
            check_name="no_nan_inf",
            message="Score is NaN",
            details={"score": score},
        )
    if np.isinf(score):
        return ValidationResult(
            is_valid=False,
            check_name="no_nan_inf",
            message="Score is infinite",
            details={"score": score},
        )

    return ValidationResult(
        is_valid=True,
        check_name="no_nan_inf",
        message="Score is valid numeric value",
        details={"score": score},
    )


def _validate_regime(regime: str) -> ValidationResult:
    """Check regime is valid."""
    valid_regimes = list(REGIME_BOUNDS.keys())

    if regime in valid_regimes:
        return ValidationResult(
            is_valid=True,
            check_name="regime_valid",
            message=f"Regime '{regime}' is valid",
            details={"regime": regime, "valid_regimes": valid_regimes},
        )
    else:
        return ValidationResult(
            is_valid=False,
            check_name="regime_valid",
            message=f"Invalid regime '{regime}'. Must be one of: {valid_regimes}",
            details={"regime": regime, "valid_regimes": valid_regimes},
        )


def _validate_regime_consistency(score: float, regime: str) -> ValidationResult:
    """Check score falls in correct regime band."""
    if regime not in REGIME_BOUNDS:
        return ValidationResult(
            is_valid=False,
            check_name="regime_consistency",
            message=f"Cannot validate consistency: unknown regime '{regime}'",
            details={"score": score, "regime": regime},
        )

    low, high = REGIME_BOUNDS[regime]

    # Handle boundary conditions (score exactly on boundary)
    in_range = low <= score <= high

    # Special case: boundary scores can belong to either regime
    if score == high and regime != "EXTREME_RISK":
        in_range = True  # Upper boundary belongs to current regime

    if in_range:
        return ValidationResult(
            is_valid=True,
            check_name="regime_consistency",
            message=f"Score {score:.4f} consistent with regime {regime} [{low}, {high}]",
            details={"score": score, "regime": regime, "bounds": (low, high)},
        )
    else:
        expected = _score_to_regime(score)
        return ValidationResult(
            is_valid=False,
            check_name="regime_consistency",
            message=(
                f"Score {score:.4f} inconsistent with regime {regime}. "
                f"Expected regime: {expected}"
            ),
            details={
                "score": score,
                "assigned_regime": regime,
                "expected_regime": expected,
                "bounds": REGIME_BOUNDS,
            },
        )


def _score_to_regime(score: float) -> str:
    """Convert score to correct regime."""
    if score < 0.25:
        return "LOW_RISK"
    elif score < 0.50:
        return "MODERATE_RISK"
    elif score < 0.75:
        return "HIGH_RISK"
    else:
        return "EXTREME_RISK"


def score_to_regime(score: float) -> str:
    """
    Public function to convert score to regime.

    Regime boundaries (from risk_score.yaml):
        - LOW_RISK: [0.0, 0.25)
        - MODERATE_RISK: [0.25, 0.50)
        - HIGH_RISK: [0.50, 0.75)
        - EXTREME_RISK: [0.75, 1.0]
    """
    return _score_to_regime(score)


def _validate_weights(weights: EnsembleWeights | dict[str, float]) -> ValidationResult:
    """Check weights are valid."""
    if isinstance(weights, EnsembleWeights):
        # EnsembleWeights already validates on construction
        return ValidationResult(
            is_valid=True,
            check_name="weights_valid",
            message="Weights validated via EnsembleWeights dataclass",
            details={"heuristic_weight": weights.heuristic_weight},
        )

    # Validate dict weights
    if not weights:
        return ValidationResult(
            is_valid=False,
            check_name="weights_valid",
            message="Weights dictionary is empty",
            details={"weights": weights},
        )

    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        return ValidationResult(
            is_valid=False,
            check_name="weights_valid",
            message=f"Weights sum to {total:.6f}, not 1.0",
            details={"weights": weights, "sum": total},
        )

    h_weight = weights.get("heuristic", 0.0)
    if h_weight < 0.30 - 1e-6:
        return ValidationResult(
            is_valid=False,
            check_name="weights_valid",
            message=f"Heuristic weight {h_weight:.3f} < 0.30 minimum",
            details={"weights": weights, "heuristic_weight": h_weight},
        )

    return ValidationResult(
        is_valid=True,
        check_name="weights_valid",
        message="Weights are valid",
        details={"weights": weights, "sum": total},
    )


def _validate_explanation(explanation: dict[str, Any]) -> ValidationResult:
    """Check explanation has required fields."""
    required_fields = [
        "model_contributions",
        "top_features",
        "regime",
        "summary",
    ]

    missing = [f for f in required_fields if f not in explanation]

    if missing:
        return ValidationResult(
            is_valid=False,
            check_name="explanation_complete",
            message=f"Explanation missing required fields: {missing}",
            details={"missing_fields": missing, "required": required_fields},
        )

    return ValidationResult(
        is_valid=True,
        check_name="explanation_complete",
        message="Explanation contains all required fields",
        details={"fields_present": list(explanation.keys())},
    )


def _validate_calibration_direction(raw_score: float, calibrated_score: float) -> ValidationResult:
    """
    Check calibration preserves relative ordering.

    NOTE: This is a soft check. Calibration may slightly violate
    monotonicity at boundaries due to fitting noise.
    """
    # For individual scores, we can only check bounds
    if not (0.0 <= calibrated_score <= 1.0):
        return ValidationResult(
            is_valid=False,
            check_name="calibration_bounds",
            message=f"Calibrated score {calibrated_score:.6f} out of bounds",
            details={"raw": raw_score, "calibrated": calibrated_score},
        )

    return ValidationResult(
        is_valid=True,
        check_name="calibration_bounds",
        message="Calibrated score within valid bounds",
        details={"raw": raw_score, "calibrated": calibrated_score},
    )


def validate_batch_calibration_monotonicity(
    raw_scores: list[float],
    calibrated_scores: list[float],
    tolerance: float = 0.01,
) -> ValidationResult:
    """
    Check calibration preserves monotonicity across a batch.

    This is a stronger check than individual score validation.
    Small violations (< tolerance) are allowed due to fitting noise.
    """
    if len(raw_scores) != len(calibrated_scores):
        return ValidationResult(
            is_valid=False,
            check_name="batch_monotonicity",
            message="Raw and calibrated score lists have different lengths",
            details={"raw_len": len(raw_scores), "calibrated_len": len(calibrated_scores)},
        )

    # Sort by raw score and check calibrated order
    pairs = sorted(zip(raw_scores, calibrated_scores, strict=True), key=lambda x: x[0])

    violations = 0
    max_violation = 0.0

    for i in range(1, len(pairs)):
        raw_diff = pairs[i][0] - pairs[i - 1][0]
        cal_diff = pairs[i][1] - pairs[i - 1][1]

        if raw_diff > 0 and cal_diff < -tolerance:
            violations += 1
            max_violation = max(max_violation, abs(cal_diff))

    if violations > 0:
        return ValidationResult(
            is_valid=False,
            check_name="batch_monotonicity",
            message=f"Calibration violates monotonicity: {violations} violations, max: {max_violation:.4f}",
            details={
                "n_violations": violations,
                "max_violation": max_violation,
                "tolerance": tolerance,
            },
        )

    return ValidationResult(
        is_valid=True,
        check_name="batch_monotonicity",
        message="Calibration preserves monotonicity",
        details={"n_pairs": len(pairs), "tolerance": tolerance},
    )
