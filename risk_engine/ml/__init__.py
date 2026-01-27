"""
Machine Learning Risk Models for MEWS-FIN.

Phase 4.1: ML-based risk scoring with SHAP explainability.

This module provides supervised learning models that map multimodal
feature snapshots to risk scores, directly comparable to the
heuristic baseline from Phase 4.0.

Model Families (STRICT - no neural networks):
    - Linear: Ridge or ElasticNet
    - Tree Ensemble: Random Forest
    - Gradient Boosting: XGBoost

Key Principles:
    - Explainability-first (SHAP mandatory)
    - Time-series splits only (no shuffling)
    - No feature leakage
    - Compare against heuristic baseline

Usage:
    from risk_engine.ml import (
        build_dataset,
        train_model,
        evaluate_model,
        compute_shap_explanations,
    )
"""

from risk_engine.ml.config import (
    ML_CONFIG,
    ModelType,
    TargetType,
)
from risk_engine.ml.dataset import (
    DatasetSplit,
    MLDataset,
    build_dataset,
    create_mock_dataset,
    create_mock_ml_dataset,
)
from risk_engine.ml.evaluate import EvaluationReport, evaluate_model
from risk_engine.ml.explain import SHAPExplanation, compute_shap_explanations
from risk_engine.ml.models import AVAILABLE_MODELS, create_model
from risk_engine.ml.train import TrainedModel, train_model

__all__ = [
    # Config
    "ML_CONFIG",
    "ModelType",
    "TargetType",
    # Dataset
    "build_dataset",
    "create_mock_dataset",
    "create_mock_ml_dataset",
    "DatasetSplit",
    "MLDataset",
    # Models
    "create_model",
    "AVAILABLE_MODELS",
    # Training
    "train_model",
    "TrainedModel",
    # Evaluation
    "evaluate_model",
    "EvaluationReport",
    # Explainability
    "compute_shap_explanations",
    "SHAPExplanation",
]
