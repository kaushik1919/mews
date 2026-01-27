"""
Model definitions for ML risk models.

SUPPORTED MODEL FAMILIES (STRICT - no neural networks):
1. Linear: Ridge, ElasticNet
2. Tree Ensemble: Random Forest
3. Gradient Boosting: XGBoost, LightGBM

All models are wrapped with a consistent interface for:
- Training
- Prediction
- Probability estimation
- Feature importance extraction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from risk_engine.ml.config import (
    ML_CONFIG,
    ModelType,
)


class BaseRiskModel(ABC):
    """
    Abstract base class for all risk models.

    Provides consistent interface across model families.
    """

    def __init__(self, model_type: ModelType, config: Any = None):
        """
        Initialize model.

        Args:
            model_type: Type of model
            config: Model-specific configuration
        """
        self.model_type = model_type
        self.config = config or ML_CONFIG.get_model_config(model_type)
        self.model = None
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.n_classes: int = 4  # Default for regime classification

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseRiskModel:
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        pass

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretation."""
        self.feature_names = names

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """
        Convert predictions to continuous risk scores [0, 1].

        For classification models, this uses expected value of
        regime probabilities, normalized to [0, 1].

        score = sum(p_i * i) / (n_classes - 1)

        Where i is the regime index (0-3) and p_i is its probability.
        """
        proba = self.predict_proba(X)
        n_classes = proba.shape[1]

        # Expected regime index
        regime_indices = np.arange(n_classes)
        expected_regime = np.dot(proba, regime_indices)

        # Normalize to [0, 1]
        risk_score = expected_regime / (n_classes - 1)
        return risk_score


class RidgeRiskModel(BaseRiskModel):
    """
    Ridge logistic regression for risk classification.

    Uses sklearn's RidgeClassifier for multi-class classification.
    Simple, interpretable, fast - good baseline.
    """

    def __init__(self, config: Any = None):
        super().__init__(ModelType.RIDGE, config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRiskModel:
        from sklearn.linear_model import RidgeClassifier

        self.model = RidgeClassifier(
            alpha=self.config.alpha,
            fit_intercept=self.config.fit_intercept,
            solver=self.config.solver,
            random_state=self.config.random_state,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # RidgeClassifier doesn't have predict_proba
        # Use decision function and softmax
        from scipy.special import softmax

        decision = self.model.decision_function(X)
        if len(decision.shape) == 1:
            # Binary case
            proba = np.column_stack([1 - decision, decision])
        else:
            proba = softmax(decision, axis=1)
        return proba

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}

        # Use absolute coefficient values as importance
        coefs = np.abs(self.model.coef_)
        if len(coefs.shape) > 1:
            # Multi-class: average across classes
            importance = coefs.mean(axis=0)
        else:
            importance = coefs

        # Normalize
        importance = importance / importance.sum()

        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance, strict=False)
        }


class ElasticNetRiskModel(BaseRiskModel):
    """
    ElasticNet logistic regression for risk classification.

    Combines L1 and L2 regularization for feature selection
    and coefficient shrinkage.
    """

    def __init__(self, config: Any = None):
        super().__init__(ModelType.ELASTICNET, config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetRiskModel:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0 / self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            fit_intercept=self.config.fit_intercept,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            multi_class="multinomial",
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}

        coefs = np.abs(self.model.coef_)
        if len(coefs.shape) > 1:
            importance = coefs.mean(axis=0)
        else:
            importance = coefs

        importance = importance / (importance.sum() + 1e-10)

        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance, strict=False)
        }


class RandomForestRiskModel(BaseRiskModel):
    """
    Random Forest classifier for risk classification.

    Tree ensemble with built-in feature importance.
    Good balance of performance and interpretability.
    """

    def __init__(self, config: Any = None):
        super().__init__(ModelType.RANDOM_FOREST, config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRiskModel:
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self.n_classes = len(np.unique(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance, strict=False)
        }


class XGBoostRiskModel(BaseRiskModel):
    """
    XGBoost classifier for risk classification.

    Gradient boosting with strong regularization.
    Often best performance among tree methods.
    """

    def __init__(self, config: Any = None):
        super().__init__(ModelType.XGBOOST, config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostRiskModel:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            ) from None

        n_classes = len(np.unique(y))
        self.n_classes = n_classes

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            use_label_encoder=False,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}

        importance = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance, strict=False)
        }


class LightGBMRiskModel(BaseRiskModel):
    """
    LightGBM classifier for risk classification.

    Fast gradient boosting with good handling of
    categorical features and missing values.
    """

    def __init__(self, config: Any = None):
        super().__init__(ModelType.LIGHTGBM, config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMRiskModel:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            ) from None

        n_classes = len(np.unique(y))
        self.n_classes = n_classes

        self.model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_samples=self.config.min_child_samples,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> dict[str, float]:
        if not self.is_fitted:
            return {}

        importance = self.model.feature_importances_
        total = importance.sum()
        if total == 0:
            total = 1.0

        return {
            name: float(imp / total)
            for name, imp in zip(self.feature_names, importance, strict=False)
        }


# Model factory
AVAILABLE_MODELS: dict[ModelType, type[BaseRiskModel]] = {
    ModelType.RIDGE: RidgeRiskModel,
    ModelType.ELASTICNET: ElasticNetRiskModel,
    ModelType.RANDOM_FOREST: RandomForestRiskModel,
    ModelType.XGBOOST: XGBoostRiskModel,
    ModelType.LIGHTGBM: LightGBMRiskModel,
}


def create_model(model_type: ModelType, config: Any = None) -> BaseRiskModel:
    """
    Factory function to create a risk model.

    Args:
        model_type: Type of model to create
        config: Model-specific configuration (uses default if None)

    Returns:
        Instantiated model (not yet fitted)

    Raises:
        ValueError: If model type not supported
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )

    model_class = AVAILABLE_MODELS[model_type]
    return model_class(config=config)
