"""
Training loop for ML risk models.

CRITICAL TRAINING RULES:
1. Time-series splits only (no shuffling)
2. No leakage across splits
3. Hyperparameters must be explicit and logged
4. Fixed random seeds for reproducibility

This is research, not AutoML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from risk_engine.ml.config import (
    ML_CONFIG,
    REGIME_LABELS,
    ModelType,
)
from risk_engine.ml.dataset import MLDataset
from risk_engine.ml.models import BaseRiskModel, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics captured during training."""

    train_accuracy: float
    val_accuracy: float
    train_loss: float | None = None
    val_loss: float | None = None
    train_class_distribution: dict[str, int] = field(default_factory=dict)
    val_class_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainedModel:
    """
    Container for a trained model with metadata.

    Includes everything needed for reproducibility:
    - The fitted model
    - Training configuration
    - Training metrics
    - Timestamps
    """

    model: BaseRiskModel
    model_type: ModelType
    config: Any
    training_metrics: TrainingMetrics
    feature_names: list[str]
    n_classes: int
    trained_at: str
    train_samples: int
    val_samples: int
    version: str = "ml-v1.0"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous risk score [0, 1]."""
        return self.model.predict_risk_score(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from model."""
        return self.model.get_feature_importance()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "version": self.version,
            "trained_at": self.trained_at,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "training_metrics": {
                "train_accuracy": self.training_metrics.train_accuracy,
                "val_accuracy": self.training_metrics.val_accuracy,
                "train_class_distribution": self.training_metrics.train_class_distribution,
                "val_class_distribution": self.training_metrics.val_class_distribution,
            },
            "feature_importance": self.get_feature_importance(),
        }


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def compute_class_distribution(y: np.ndarray) -> dict[str, int]:
    """Compute class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    return {
        REGIME_LABELS.get(int(u), str(u)): int(c)
        for u, c in zip(unique, counts, strict=False)
    }


def train_model(
    model_type: ModelType,
    dataset: MLDataset,
    config: Any = None,
    verbose: bool = True,
) -> TrainedModel:
    """
    Train a single model on the dataset.

    Args:
        model_type: Type of model to train
        dataset: MLDataset with train/val/test splits
        config: Model-specific config (uses default if None)
        verbose: Print training progress

    Returns:
        TrainedModel with fitted model and metadata

    CRITICAL:
    - No shuffling of training data
    - Validation set used only for monitoring, not early stopping
    - All hyperparameters are fixed before training
    """
    if config is None:
        config = ML_CONFIG.get_model_config(model_type)

    if verbose:
        logger.info(f"Training {model_type.value} model...")
        logger.info(f"  Train samples: {dataset.train.n_samples}")
        logger.info(f"  Val samples: {dataset.val.n_samples}")
        logger.info(f"  Features: {dataset.train.n_features}")

    # Create model
    model = create_model(model_type, config)
    model.set_feature_names(dataset.feature_names)

    # Fit on training data
    model.fit(dataset.train.X, dataset.train.y)

    # Compute training metrics
    train_preds = model.predict(dataset.train.X)
    val_preds = model.predict(dataset.val.X)

    train_accuracy = compute_accuracy(dataset.train.y, train_preds)
    val_accuracy = compute_accuracy(dataset.val.y, val_preds)

    train_distribution = compute_class_distribution(dataset.train.y)
    val_distribution = compute_class_distribution(dataset.val.y)

    metrics = TrainingMetrics(
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        train_class_distribution=train_distribution,
        val_class_distribution=val_distribution,
    )

    if verbose:
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Val accuracy: {val_accuracy:.4f}")
        logger.info(f"  Train distribution: {train_distribution}")

    trained = TrainedModel(
        model=model,
        model_type=model_type,
        config=config,
        training_metrics=metrics,
        feature_names=dataset.feature_names,
        n_classes=model.n_classes,
        trained_at=datetime.utcnow().isoformat() + "Z",
        train_samples=dataset.train.n_samples,
        val_samples=dataset.val.n_samples,
    )

    return trained


def train_all_models(
    dataset: MLDataset,
    model_types: list[ModelType] | None = None,
    verbose: bool = True,
) -> dict[ModelType, TrainedModel]:
    """
    Train all specified model types on the dataset.

    Args:
        dataset: MLDataset with train/val/test splits
        model_types: List of model types to train (default: all)
        verbose: Print training progress

    Returns:
        Dict of model_type -> TrainedModel
    """
    if model_types is None:
        # Default: one from each family
        model_types = [
            ModelType.RIDGE,
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
        ]

    trained_models: dict[ModelType, TrainedModel] = {}

    for model_type in model_types:
        try:
            trained = train_model(model_type, dataset, verbose=verbose)
            trained_models[model_type] = trained
        except ImportError as e:
            logger.warning(f"Skipping {model_type.value}: {e}")
        except Exception as e:
            logger.error(f"Failed to train {model_type.value}: {e}")
            raise

    return trained_models


def cross_validate_time_series(
    model_type: ModelType,
    feature_data: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    config: Any = None,
) -> list[float]:
    """
    Time-series cross-validation with expanding window.

    CRITICAL: No shuffling. Each fold uses only past data for training.

    Args:
        model_type: Type of model to train
        feature_data: Feature matrix
        labels: Target labels
        n_splits: Number of CV folds
        test_size: Fraction of data for each test fold
        config: Model configuration

    Returns:
        List of validation accuracies for each fold
    """
    n_samples = len(labels)
    fold_size = int(n_samples * test_size)

    scores = []

    for i in range(n_splits):
        # Training: all data before test fold
        # Test: next fold_size samples
        test_start = n_samples - (n_splits - i) * fold_size
        test_end = test_start + fold_size

        if test_start < fold_size:
            # Not enough training data
            continue

        train_X = feature_data[:test_start]
        train_y = labels[:test_start]
        test_X = feature_data[test_start:test_end]
        test_y = labels[test_start:test_end]

        # Train model
        model = create_model(model_type, config)
        model.fit(train_X, train_y)

        # Evaluate
        preds = model.predict(test_X)
        accuracy = compute_accuracy(test_y, preds)
        scores.append(accuracy)

    return scores
