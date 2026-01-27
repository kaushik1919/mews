"""
Ensemble & Calibration Layer for MEWS-FIN.

Phase 4.2: Combines heuristic + ML model outputs into a single
calibrated, stable risk score that behaves well across regimes and time.

This is the DECISION-GRADE output layer.

Key Principles:
    - Heuristic anchors interpretability (weight ≥ 0.30)
    - ML adds pattern detection (weight ≤ 0.70)
    - Calibration ensures probability semantics
    - Smoothing enhances stability without masking spikes
    - Explainability is mandatory

Usage:
    from risk_engine.ensemble import compute_ensemble_risk

    result = compute_ensemble_risk(
        heuristic_score=heuristic_snapshot,
        ml_scores={"random_forest": rf_snapshot},
        as_of=pd.Timestamp("2024-01-15"),
    )

    print(result.risk_score)  # 0.62
    print(result.regime)      # "HIGH_RISK"
    print(result.model_contributions)  # {"heuristic": 0.35, "random_forest": 0.27}
"""

from risk_engine.ensemble.calibration import (
    CalibrationMethod,
    CalibratorConfig,
    FittedCalibrator,
    apply_calibration,
    fit_calibrator,
)
from risk_engine.ensemble.explain import (
    EnsembleExplanation,
    FeatureContribution,
    ModelContribution,
    compute_ensemble_explanation,
)
from risk_engine.ensemble.service import (
    ENSEMBLE_CONFIG,
    EnsembleConfig,
    EnsembleRiskSnapshot,
    compute_ensemble_risk,
    compute_ensemble_risk_from_scores,
)
from risk_engine.ensemble.smoothing import (
    SmoothingConfig,
    SmoothingMethod,
    SmoothingState,
    apply_temporal_smoothing,
)
from risk_engine.ensemble.validate import (
    ValidationReport,
    ValidationResult,
    score_to_regime,
    validate_ensemble_output,
)
from risk_engine.ensemble.weights import (
    ENSEMBLE_WEIGHTS,
    EnsembleWeights,
    ModelRole,
    compute_weighted_ensemble,
)

__all__ = [
    # Config
    "EnsembleConfig",
    "ENSEMBLE_CONFIG",
    "EnsembleWeights",
    "ENSEMBLE_WEIGHTS",
    "ModelRole",
    # Calibration
    "CalibrationMethod",
    "CalibratorConfig",
    "FittedCalibrator",
    "fit_calibrator",
    "apply_calibration",
    # Smoothing
    "SmoothingConfig",
    "SmoothingState",
    "SmoothingMethod",
    "apply_temporal_smoothing",
    # Explanation
    "EnsembleExplanation",
    "ModelContribution",
    "FeatureContribution",
    "compute_ensemble_explanation",
    # Validation
    "validate_ensemble_output",
    "score_to_regime",
    "ValidationReport",
    "ValidationResult",
    # Weights
    "compute_weighted_ensemble",
    # Main API
    "EnsembleRiskSnapshot",
    "compute_ensemble_risk",
    "compute_ensemble_risk_from_scores",
]
