"""
Calibration Module for Ensemble Risk Scores.

MEWS-FIN Phase 4.2: Transform raw ensemble scores into calibrated
probabilities with proper probability semantics.

Supported Methods:
    - Platt Scaling: Logistic regression calibration (smooth)
    - Isotonic Regression: Non-parametric, monotonic (flexible)

Key Constraints:
    - Calibration MUST be fit on train + validation data ONLY
    - Calibration MUST be applied to test / future data
    - Calibration MUST be deterministic
    - NO data leakage from test set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Supported calibration methods."""

    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    NONE = "none"  # Pass-through (no calibration)


@dataclass
class CalibratorConfig:
    """Configuration for calibration."""

    method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION

    # Platt scaling parameters
    platt_regularization: float = 1.0
    platt_max_iter: int = 1000

    # Isotonic regression parameters
    isotonic_out_of_bounds: str = "clip"  # "clip" or "nan"

    # Validation
    check_monotonicity: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "platt_regularization": self.platt_regularization,
            "platt_max_iter": self.platt_max_iter,
            "isotonic_out_of_bounds": self.isotonic_out_of_bounds,
            "check_monotonicity": self.check_monotonicity,
        }


@dataclass
class FittedCalibrator:
    """
    A fitted calibrator that can transform raw scores.

    Attributes:
        method: Calibration method used
        fitted_params: Method-specific fitted parameters
        train_score_range: (min, max) of training scores for validation
        is_fitted: Whether the calibrator has been fitted
    """

    method: CalibrationMethod
    fitted_params: dict[str, Any] = field(default_factory=dict)
    train_score_range: tuple[float, float] = (0.0, 1.0)
    is_fitted: bool = False

    # Internal model storage
    _platt_slope: float = 1.0
    _platt_intercept: float = 0.0
    _isotonic_x: np.ndarray | None = None
    _isotonic_y: np.ndarray | None = None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw scores.

        Args:
            scores: Raw ensemble scores in [0, 1]

        Returns:
            Calibrated scores in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")

        if self.method == CalibrationMethod.NONE:
            return scores

        elif self.method == CalibrationMethod.PLATT_SCALING:
            return self._platt_transform(scores)

        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            return self._isotonic_transform(scores)

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

    def _platt_transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        # Logistic transformation
        logits = self._platt_slope * scores + self._platt_intercept
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(calibrated, 0.0, 1.0)

    def _isotonic_transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic regression calibration."""
        if self._isotonic_x is None or self._isotonic_y is None:
            raise RuntimeError("Isotonic calibrator not properly fitted")

        # Use linear interpolation between fitted points
        calibrated = np.interp(
            scores,
            self._isotonic_x,
            self._isotonic_y,
        )
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted calibrator."""
        result = {
            "method": self.method.value,
            "train_score_range": list(self.train_score_range),
            "is_fitted": self.is_fitted,
        }

        if self.method == CalibrationMethod.PLATT_SCALING:
            result["platt_slope"] = self._platt_slope
            result["platt_intercept"] = self._platt_intercept

        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            if self._isotonic_x is not None:
                result["isotonic_x"] = self._isotonic_x.tolist()
                result["isotonic_y"] = self._isotonic_y.tolist()

        return result


def fit_calibrator(
    *,
    raw_scores: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    y: np.ndarray | None = None,
    method: str | CalibrationMethod | CalibratorConfig | None = "platt",
) -> FittedCalibrator:
    """
    Fit a calibrator on training + validation data.

    CRITICAL: This must ONLY be called with train + val data.
    Never include test data in calibration fitting.

    Args:
        raw_scores: Raw ensemble scores from train + val (alias for scores)
        scores: Raw ensemble scores from train + val
        targets: True regime labels (0-3) or binary labels
        y: Alias for targets (backward compatibility)
        method: Calibration method or configuration

    Returns:
        Fitted calibrator ready for transform()

    Raises:
        ValueError: If inputs are invalid
    """
    # Resolve scores (accept either raw_scores or scores, not both)
    if raw_scores is not None and scores is not None:
        raise ValueError("Provide only one of `raw_scores` or `scores`")

    if raw_scores is None and scores is None:
        raise ValueError("One of `raw_scores` or `scores` must be provided")

    if scores is None:
        scores = raw_scores

    # Resolve targets (accept either targets or y, not both)
    if targets is not None and y is not None:
        raise ValueError("Provide only one of `targets` or `y`")

    if targets is None and y is None:
        raise ValueError("One of `targets` or `y` must be provided")

    if targets is None:
        targets = y

    # Handle convenience usage: method passed directly
    config: CalibratorConfig
    if isinstance(method, CalibratorConfig):
        config = method
    elif isinstance(method, CalibrationMethod):
        config = CalibratorConfig(method=method)
    elif isinstance(method, str):
        # Map string to CalibrationMethod
        method_map = {
            "platt": CalibrationMethod.PLATT_SCALING,
            "platt_scaling": CalibrationMethod.PLATT_SCALING,
            "isotonic": CalibrationMethod.ISOTONIC_REGRESSION,
            "isotonic_regression": CalibrationMethod.ISOTONIC_REGRESSION,
            "none": CalibrationMethod.NONE,
        }
        cal_method = method_map.get(method.lower(), CalibrationMethod.ISOTONIC_REGRESSION)
        config = CalibratorConfig(method=cal_method)
    elif method is None:
        config = CalibratorConfig()
    else:
        config = CalibratorConfig()

    # Use canonical names internally (scores and targets only from here)
    raw_scores_arr = np.asarray(scores).flatten()
    true_labels = np.asarray(targets).flatten()

    if len(raw_scores_arr) != len(true_labels):
        raise ValueError(
            f"Score and label lengths must match: {len(raw_scores_arr)} vs {len(true_labels)}"
        )

    if len(raw_scores_arr) < 10:
        logger.warning(
            f"Very few samples for calibration: {len(raw_scores_arr)}. "
            "Consider using NONE method."
        )

    # Create calibrator
    calibrator = FittedCalibrator(
        method=config.method,
        train_score_range=(float(raw_scores_arr.min()), float(raw_scores_arr.max())),
    )

    if config.method == CalibrationMethod.NONE:
        calibrator.is_fitted = True
        return calibrator

    # Convert labels to probabilities for calibration
    # For regime classification, normalize to [0, 1]
    n_classes = int(true_labels.max()) + 1
    target_probs = true_labels / (n_classes - 1) if n_classes > 1 else true_labels

    if config.method == CalibrationMethod.PLATT_SCALING:
        _fit_platt(calibrator, raw_scores_arr, target_probs, config)

    elif config.method == CalibrationMethod.ISOTONIC_REGRESSION:
        _fit_isotonic(calibrator, raw_scores_arr, target_probs, config)

    calibrator.is_fitted = True
    return calibrator


def _fit_platt(
    calibrator: FittedCalibrator,
    raw_scores: np.ndarray,
    target_probs: np.ndarray,
    config: CalibratorConfig,
) -> None:
    """
    Fit Platt scaling parameters.

    Platt scaling fits a logistic regression:
    P(y=1|s) = 1 / (1 + exp(A*s + B))

    We optimize A and B to minimize cross-entropy.
    """
    from scipy.optimize import minimize

    def neg_log_likelihood(params: np.ndarray) -> float:
        """Negative log-likelihood for Platt scaling."""
        slope, intercept = params
        logits = slope * raw_scores + intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # Cross-entropy loss
        loss = -np.mean(
            target_probs * np.log(probs) +
            (1 - target_probs) * np.log(1 - probs)
        )
        # L2 regularization
        loss += config.platt_regularization * (slope ** 2 + intercept ** 2)
        return loss

    # Initial guess
    x0 = np.array([1.0, 0.0])

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": config.platt_max_iter},
    )

    calibrator._platt_slope = float(result.x[0])
    calibrator._platt_intercept = float(result.x[1])
    calibrator.fitted_params = {
        "slope": calibrator._platt_slope,
        "intercept": calibrator._platt_intercept,
        "optimization_success": result.success,
    }


def _fit_isotonic(
    calibrator: FittedCalibrator,
    raw_scores: np.ndarray,
    target_probs: np.ndarray,
    config: CalibratorConfig,
) -> None:
    """
    Fit isotonic regression.

    Isotonic regression finds a monotonically increasing function
    that minimizes squared error.
    """

    # Sort by raw scores
    sort_idx = np.argsort(raw_scores)
    sorted_scores = raw_scores[sort_idx]
    sorted_targets = target_probs[sort_idx]

    # Pool Adjacent Violators Algorithm (PAVA)
    calibrated = _pava(sorted_targets)

    # Store as interpolation points
    # Remove duplicates by taking unique score values
    unique_scores, unique_idx = np.unique(sorted_scores, return_index=True)
    unique_calibrated = calibrated[unique_idx]

    # Ensure we have endpoints at 0 and 1
    if unique_scores[0] > 0:
        unique_scores = np.concatenate([[0.0], unique_scores])
        unique_calibrated = np.concatenate([[unique_calibrated[0]], unique_calibrated])
    if unique_scores[-1] < 1:
        unique_scores = np.concatenate([unique_scores, [1.0]])
        unique_calibrated = np.concatenate([unique_calibrated, [unique_calibrated[-1]]])

    calibrator._isotonic_x = unique_scores
    calibrator._isotonic_y = unique_calibrated
    calibrator.fitted_params = {
        "n_unique_points": len(unique_scores),
        "output_range": (float(unique_calibrated.min()), float(unique_calibrated.max())),
    }

    # Validate monotonicity
    if config.check_monotonicity:
        if not np.all(np.diff(unique_calibrated) >= -1e-10):
            logger.warning("Isotonic calibration is not strictly monotonic")


def _pava(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm for isotonic regression.

    Ensures output is monotonically non-decreasing.

    Args:
        y: Target values (sorted by x)

    Returns:
        Isotonic fit of y
    """
    n = len(y)
    result = y.copy()

    # Pool adjacent violators
    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Find block of violators
            j = i + 1
            while j < n - 1 and result[j] > result[j + 1]:
                j += 1

            # Pool the block [i, j]
            block_mean = np.mean(result[i:j + 1])
            result[i:j + 1] = block_mean

            # Step back to check previous blocks
            if i > 0:
                i -= 1
            else:
                i = j + 1
        else:
            i += 1

    return result


def apply_calibration(
    raw_score: float | np.ndarray,
    calibrator: FittedCalibrator,
) -> float | np.ndarray:
    """
    Apply calibration to raw score(s).

    Convenience function for single-score calibration.

    Args:
        raw_score: Raw ensemble score(s) in [0, 1]
        calibrator: Fitted calibrator

    Returns:
        Calibrated score(s) in [0, 1]
    """
    is_scalar = np.isscalar(raw_score)
    scores = np.atleast_1d(raw_score).astype(float)

    calibrated = calibrator.transform(scores)

    if is_scalar:
        return float(calibrated[0])
    return calibrated


def create_identity_calibrator() -> FittedCalibrator:
    """
    Create a pass-through calibrator that doesn't change scores.

    Useful for testing or when calibration is disabled.
    """
    calibrator = FittedCalibrator(method=CalibrationMethod.NONE)
    calibrator.is_fitted = True
    return calibrator
