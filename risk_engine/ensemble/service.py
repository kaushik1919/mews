"""
Ensemble Risk Engine Public API.

MEWS-FIN Phase 4.2: Combine heuristic and ML model outputs into a
single calibrated, stable risk score that behaves well across regimes and time.

This is where MEWS-FIN becomes decision-grade.

Usage:
    from risk_engine.ensemble import compute_ensemble_risk

    result = compute_ensemble_risk(
        heuristic_snapshot=heuristic_result,
        ml_models={"random_forest": rf_model, "xgboost": xgb_model},
        features=feature_dict,
        as_of=pd.Timestamp("2024-01-15T16:00:00Z"),
        calibrator=fitted_calibrator,  # Optional
        smoothing_state=state,  # Optional
    )

    print(result.risk_score)     # 0.58
    print(result.regime)         # "HIGH_RISK"
    print(result.explanation)    # Full explainability

Key Guarantees:
    - Heuristic weight >= 0.30 (interpretability anchor)
    - ML weight <= 0.70 (data-driven signal)
    - Calibrated probabilities (if calibrator provided)
    - Temporal stability (if smoothing enabled)
    - Full explainability (always)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from risk_engine.heuristic.service import RiskScoreSnapshot as HeuristicSnapshot
from risk_engine.ml.train import TrainedModel

from .calibration import (
    CalibrationMethod,
    FittedCalibrator,
    apply_calibration,
)
from .explain import (
    EnsembleExplanation,
    compute_ensemble_explanation,
)
from .smoothing import (
    SmoothingConfig,
    SmoothingState,
    apply_temporal_smoothing,
)
from .validate import (
    ValidationReport,
    score_to_regime,
    validate_ensemble_output,
)
from .weights import (
    ENSEMBLE_WEIGHTS,
    EnsembleWeights,
    compute_weighted_ensemble,
)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble risk scoring.

    Attributes:
        weights: Ensemble weights for each model
        calibration_method: Calibration method to use
        smoothing_config: Temporal smoothing configuration
        require_all_models: If True, fail if any model is missing
        n_top_features: Number of top features in explanation
    """

    weights: EnsembleWeights = field(default_factory=lambda: ENSEMBLE_WEIGHTS)
    calibration_method: CalibrationMethod = CalibrationMethod.PLATT_SCALING
    smoothing_config: SmoothingConfig = field(default_factory=SmoothingConfig)
    require_all_models: bool = False
    n_top_features: int = 5


# Default configuration
ENSEMBLE_CONFIG = EnsembleConfig()


@dataclass
class EnsembleRiskSnapshot:
    """
    Complete ensemble risk score output with full explainability.

    This is the primary output of the ensemble risk engine.
    All fields follow core-specs/risk_score.yaml contract.

    Attributes:
        risk_score: Final calibrated, smoothed risk score in [0, 1]
        regime: Risk regime: LOW_RISK, MODERATE_RISK, HIGH_RISK, EXTREME_RISK
        as_of: Timestamp of the snapshot
        raw_ensemble_score: Score before calibration/smoothing
        calibrated_score: Score after calibration, before smoothing
        smoothed_score: Score after smoothing (same as risk_score if smoothing enabled)

        model_scores: Dict of model_name -> raw score
        model_contributions: Dict of model_name -> weighted contribution

        explanation: Full ensemble explanation
        validation: Validation report

        warnings: List of warnings
        version: Engine version
    """

    risk_score: float
    regime: str
    as_of: str

    # Score decomposition
    raw_ensemble_score: float
    calibrated_score: float
    smoothed_score: float

    # Model contributions
    model_scores: dict[str, float]
    model_contributions: dict[str, float]

    # Explainability
    explanation: EnsembleExplanation
    validation: ValidationReport

    # Metadata
    warnings: list[str] = field(default_factory=list)
    version: str = "ensemble-v1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "risk_score": self.risk_score,
            "regime": self.regime,
            "as_of": self.as_of,
            "raw_ensemble_score": self.raw_ensemble_score,
            "calibrated_score": self.calibrated_score,
            "smoothed_score": self.smoothed_score,
            "model_scores": self.model_scores,
            "model_contributions": self.model_contributions,
            "explanation": self.explanation.to_dict(),
            "validation": self.validation.to_dict(),
            "warnings": self.warnings,
            "version": self.version,
        }


def compute_ensemble_risk(
    heuristic_snapshot: HeuristicSnapshot,
    ml_models: dict[str, TrainedModel] | None = None,
    features: dict[str, float] | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    calibrator: FittedCalibrator | None = None,
    smoothing_state: SmoothingState | None = None,
    config: EnsembleConfig | None = None,
) -> EnsembleRiskSnapshot:
    """
    Compute ensemble risk score from heuristic and ML models.

    This is the main entry point for the ensemble risk engine.

    Args:
        heuristic_snapshot: RiskScoreSnapshot from heuristic engine
        ml_models: Dict of model_name -> TrainedModel (optional)
        features: Feature dict for ML prediction (required if ml_models provided)
        as_of: Timestamp of the snapshot. Defaults to heuristic snapshot's as_of.
        calibrator: Fitted calibrator for probability calibration (optional)
        smoothing_state: State for temporal smoothing (optional)
        config: Ensemble configuration (optional, uses defaults)

    Returns:
        Complete ensemble risk snapshot with explainability

    Raises:
        ValueError: If validation fails
    """
    config = config or ENSEMBLE_CONFIG
    warnings: list[str] = []

    # Parse timestamp
    if as_of is None:
        as_of_str = heuristic_snapshot.as_of
    elif isinstance(as_of, str):
        as_of_str = as_of
    elif isinstance(as_of, datetime):
        as_of_str = as_of.isoformat()
    elif isinstance(as_of, pd.Timestamp):
        as_of_str = as_of.isoformat()
    else:
        as_of_str = str(as_of)

    # Get heuristic score
    heuristic_score = heuristic_snapshot.risk_score
    if heuristic_score is None:
        raise ValueError("Heuristic snapshot has no risk score")

    # Get ML scores
    ml_scores: dict[str, float] = {}
    ml_shap_values: dict[str, float] | None = None

    if ml_models and features:
        ml_scores, ml_shap_values = _compute_ml_scores(
            ml_models, features, config.weights
        )
    elif ml_models and not features:
        warnings.append("ML models provided but no features; using heuristic only")
    elif not ml_models:
        warnings.append("No ML models provided; using heuristic only")

    # Compute weighted ensemble
    if ml_scores:
        raw_ensemble, contributions = compute_weighted_ensemble(
            heuristic_score=heuristic_score,
            ml_scores=ml_scores,
            weights=config.weights,
        )
        model_scores = {"heuristic": heuristic_score, **ml_scores}
    else:
        # Heuristic-only mode
        raw_ensemble = heuristic_score
        contributions = {"heuristic": heuristic_score}
        model_scores = {"heuristic": heuristic_score}

    # Apply calibration
    if calibrator is not None:
        calibrated = apply_calibration(raw_ensemble, calibrator)
    else:
        calibrated = raw_ensemble

    # Apply temporal smoothing
    if smoothing_state is not None:
        smoothed = apply_temporal_smoothing(
            current_score=calibrated,
            state=smoothing_state,
            config=config.smoothing_config,
        )
    else:
        smoothed = calibrated

    # Final risk score
    final_score = smoothed

    # Determine regime
    regime = score_to_regime(final_score)

    # Compute weights dict for explanation
    weights_dict = _weights_to_dict(config.weights)

    # Generate explanation
    explanation = compute_ensemble_explanation(
        ensemble_score=final_score,
        heuristic_score=heuristic_score,
        heuristic_contributions=heuristic_snapshot.feature_contributions,
        ml_scores=ml_scores,
        ml_shap_values=ml_shap_values,
        model_weights=weights_dict,
        regime=regime,
        n_top_features=config.n_top_features,
    )

    # Validate output
    validation = validate_ensemble_output(
        score=final_score,
        regime=regime,
        weights=config.weights,
        explanation=explanation.to_dict(),
        calibrated=calibrator is not None,
        raw_score=raw_ensemble if calibrator is not None else None,
    )

    # Add validation warnings
    if not validation.is_valid:
        for failure in validation.critical_failures:
            warnings.append(f"VALIDATION: {failure}")

    return EnsembleRiskSnapshot(
        risk_score=final_score,
        regime=regime,
        as_of=as_of_str,
        raw_ensemble_score=raw_ensemble,
        calibrated_score=calibrated,
        smoothed_score=smoothed,
        model_scores=model_scores,
        model_contributions=contributions,
        explanation=explanation,
        validation=validation,
        warnings=warnings,
        version="ensemble-v1.0",
    )


def _compute_ml_scores(
    ml_models: dict[str, TrainedModel],
    features: dict[str, float],
    weights: EnsembleWeights,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """
    Compute risk scores from ML models.

    Returns:
        Tuple of (model_scores, shap_values_from_primary_model)
    """
    ml_scores: dict[str, float] = {}
    primary_shap: dict[str, float] | None = None

    # Determine which model is primary for SHAP
    primary_model_name = weights.primary_ml_model

    for model_name, model in ml_models.items():
        # Prepare feature array
        feature_names = model.feature_names
        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        # Predict risk score
        risk_score = model.predict_risk_score(X)[0]
        ml_scores[model_name] = float(risk_score)

        # Get SHAP values from primary model
        if model_name == primary_model_name:
            try:
                importance = model.get_feature_importance()
                primary_shap = importance
            except (AttributeError, NotImplementedError):
                primary_shap = None

    return ml_scores, primary_shap


def _weights_to_dict(weights: EnsembleWeights) -> dict[str, float]:
    """Convert EnsembleWeights to dict for explanation."""
    result = {"heuristic": weights.heuristic_weight}

    if weights.primary_ml_model and weights.primary_ml_weight > 0:
        result[weights.primary_ml_model] = weights.primary_ml_weight

    if weights.secondary_ml_model and weights.secondary_ml_weight > 0:
        result[weights.secondary_ml_model] = weights.secondary_ml_weight

    return result


def compute_ensemble_risk_from_scores(
    heuristic_score: float,
    ml_scores: dict[str, float] | None = None,
    heuristic_contributions: dict[str, float] | None = None,
    ml_shap_values: dict[str, float] | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
    calibrator: FittedCalibrator | None = None,
    smoothing_state: SmoothingState | None = None,
    config: EnsembleConfig | None = None,
) -> EnsembleRiskSnapshot:
    """
    Compute ensemble risk score from pre-computed scores.

    Simpler interface when you already have model scores computed.

    Args:
        heuristic_score: Risk score from heuristic engine [0, 1]
        ml_scores: Dict of model_name -> score (optional)
        heuristic_contributions: Feature contributions from heuristic
        ml_shap_values: SHAP values from primary ML model
        as_of: Timestamp of the snapshot
        calibrator: Fitted calibrator (optional)
        smoothing_state: State for temporal smoothing (optional)
        config: Ensemble configuration (optional)

    Returns:
        Complete ensemble risk snapshot
    """
    config = config or ENSEMBLE_CONFIG
    warnings: list[str] = []
    ml_scores = ml_scores or {}
    heuristic_contributions = heuristic_contributions or {}

    # Parse timestamp
    if as_of is None:
        as_of_str = datetime.utcnow().isoformat()
    elif isinstance(as_of, str):
        as_of_str = as_of
    elif isinstance(as_of, datetime):
        as_of_str = as_of.isoformat()
    elif isinstance(as_of, pd.Timestamp):
        as_of_str = as_of.isoformat()
    else:
        as_of_str = str(as_of)

    # Compute weighted ensemble
    if ml_scores:
        raw_ensemble, contributions = compute_weighted_ensemble(
            heuristic_score=heuristic_score,
            ml_scores=ml_scores,
            weights=config.weights,
        )
        model_scores = {"heuristic": heuristic_score, **ml_scores}
    else:
        raw_ensemble = heuristic_score
        contributions = {"heuristic": heuristic_score}
        model_scores = {"heuristic": heuristic_score}

    # Apply calibration
    if calibrator is not None:
        calibrated = apply_calibration(raw_ensemble, calibrator)
    else:
        calibrated = raw_ensemble

    # Apply temporal smoothing
    if smoothing_state is not None:
        smoothed = apply_temporal_smoothing(
            current_score=calibrated,
            state=smoothing_state,
            config=config.smoothing_config,
        )
    else:
        smoothed = calibrated

    final_score = smoothed
    regime = score_to_regime(final_score)

    # Generate explanation
    weights_dict = _weights_to_dict(config.weights)
    explanation = compute_ensemble_explanation(
        ensemble_score=final_score,
        heuristic_score=heuristic_score,
        heuristic_contributions=heuristic_contributions,
        ml_scores=ml_scores,
        ml_shap_values=ml_shap_values,
        model_weights=weights_dict,
        regime=regime,
        n_top_features=config.n_top_features,
    )

    # Validate
    validation = validate_ensemble_output(
        score=final_score,
        regime=regime,
        weights=config.weights,
        explanation=explanation.to_dict(),
        calibrated=calibrator is not None,
        raw_score=raw_ensemble if calibrator is not None else None,
    )

    return EnsembleRiskSnapshot(
        risk_score=final_score,
        regime=regime,
        as_of=as_of_str,
        raw_ensemble_score=raw_ensemble,
        calibrated_score=calibrated,
        smoothed_score=smoothed,
        model_scores=model_scores,
        model_contributions=contributions,
        explanation=explanation,
        validation=validation,
        warnings=warnings,
        version="ensemble-v1.0",
    )
