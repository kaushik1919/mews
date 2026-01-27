"""
Configuration for ML risk models.

CRITICAL DECISIONS DOCUMENTED HERE:

TARGET VARIABLE CHOICE: Option A (Regime Classification)
---------------------------------------------------------
We choose multi-class regime classification over continuous targets
because:

1. INTERPRETABILITY: Regime labels (Low/Moderate/High/Extreme) directly
   map to risk_score.yaml semantic bands, making model outputs
   immediately comparable to heuristic baseline.

2. ACTIONABILITY: Risk committees care about regime transitions, not
   precise score values. "Are we in High Risk?" is the key question.

3. CALIBRATION: Historical anchors in risk_score.yaml define expected
   score ranges for known crisis periods, providing natural labels.

4. ROBUSTNESS: Classification is more robust to label noise than
   regression on synthetic continuous targets.

Label Construction:
    - Use historical VIX + realized volatility + drawdown to define regimes
    - Validate against risk_score.yaml historical anchors
    - 4 classes: LOW_RISK, MODERATE_RISK, HIGH_RISK, EXTREME_RISK

Alternative (Option B) would use forward realized volatility as target,
but this introduces:
    - Arbitrary horizon choice
    - Lookahead in label (must handle carefully)
    - Less interpretable outputs

HYPERPARAMETER PHILOSOPHY:
- All hyperparameters explicit and fixed
- No AutoML or hyperparameter search in Phase 4.1
- Focus on interpretability, not marginal performance gains
- Hyperparameters chosen for stability, not optimality
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TargetType(Enum):
    """Target variable type for ML models."""

    REGIME_CLASSIFICATION = "regime_classification"  # 4-class: Low/Mod/High/Extreme
    BINARY_CRISIS = "binary_crisis"  # 2-class: Normal vs Crisis (High+Extreme)
    FORWARD_VOLATILITY = "forward_volatility"  # Continuous: 30d forward vol


class ModelType(Enum):
    """Supported model families."""

    RIDGE = "ridge"
    ELASTICNET = "elasticnet"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


@dataclass
class TimeSeriesSplitConfig:
    """
    Time-based train/validation/test split configuration.

    CRITICAL: No random splits. Must respect temporal order.
    """

    train_start: str = "2005-01-01"
    train_end: str = "2014-12-31"
    val_start: str = "2015-01-01"
    val_end: str = "2018-12-31"
    test_start: str = "2019-01-01"
    test_end: str = "2024-12-31"

    # Gap between train/val and val/test to avoid leakage
    # (e.g., if using forward-looking labels)
    gap_days: int = 0  # Set to 30 if using forward vol target


@dataclass
class RidgeConfig:
    """Ridge regression hyperparameters."""

    alpha: float = 1.0
    fit_intercept: bool = True
    solver: str = "auto"
    random_state: int = 42


@dataclass
class ElasticNetConfig:
    """ElasticNet hyperparameters."""

    alpha: float = 1.0
    l1_ratio: float = 0.5  # Balance between L1 and L2
    fit_intercept: bool = True
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters."""

    n_estimators: int = 100
    max_depth: int | None = 10  # Limit depth for interpretability
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    class_weight: str = "balanced"  # Handle class imbalance
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    objective: str = "multi:softprob"  # For classification
    eval_metric: str = "mlogloss"
    use_label_encoder: bool = False
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


@dataclass
class SHAPConfig:
    """SHAP explainability configuration."""

    # Maximum samples for SHAP computation (for performance)
    max_samples: int = 1000

    # Background samples for SHAP explainer
    background_samples: int = 100

    # Top features to report
    top_features: int = 10

    # Compute per-sample explanations for test set
    compute_local_explanations: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    # Crisis periods for targeted evaluation
    crisis_periods: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("2008-09-01", "2009-03-31", "2008 GFC"),
            ("2011-07-01", "2011-12-31", "2011 Euro Crisis"),
            ("2015-08-01", "2015-09-30", "2015 China"),
            ("2018-10-01", "2019-01-31", "2018 Q4 Selloff"),
            ("2020-02-01", "2020-04-30", "2020 COVID"),
            ("2022-01-01", "2022-10-31", "2022 Rate Hike"),
        ]
    )

    # Lead time analysis: how many days before crisis did model elevate risk?
    lead_time_windows: list[int] = field(
        default_factory=lambda: [5, 10, 20, 30]
    )


@dataclass
class MLConfig:
    """
    Master configuration for ML risk models.

    All settings are explicit and documented.
    No hidden defaults or AutoML.
    """

    # Target variable choice (CRITICAL DECISION)
    target_type: TargetType = TargetType.REGIME_CLASSIFICATION

    # For forward volatility target: prediction horizon
    forward_horizon_days: int = 30

    # Time-series splits
    splits: TimeSeriesSplitConfig = field(default_factory=TimeSeriesSplitConfig)

    # Model hyperparameters
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    elasticnet: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)

    # SHAP configuration
    shap: SHAPConfig = field(default_factory=SHAPConfig)

    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Feature handling
    missing_value_strategy: str = "impute_median"  # or "drop"
    standardize_features: bool = True

    # Reproducibility
    random_state: int = 42

    def get_model_config(self, model_type: ModelType) -> Any:
        """Get configuration for specific model type."""
        config_map = {
            ModelType.RIDGE: self.ridge,
            ModelType.ELASTICNET: self.elasticnet,
            ModelType.RANDOM_FOREST: self.random_forest,
            ModelType.XGBOOST: self.xgboost,
            ModelType.LIGHTGBM: self.lightgbm,
        }
        return config_map.get(model_type)


# Global configuration instance
ML_CONFIG = MLConfig()


# Feature names from Phase 3 (must match exactly)
NUMERIC_FEATURES = [
    "realized_volatility_20d",
    "realized_volatility_60d",
    "volatility_ratio_20d_60d",
    "max_drawdown_20d",
    "max_drawdown_60d",
    "volume_zscore_20d",
    "volume_price_divergence",
    "vix_level",
]

SENTIMENT_FEATURES = [
    "news_sentiment_daily",
    "news_sentiment_5d",
    "sentiment_volatility_20d",
]

GRAPH_FEATURES = [
    "avg_pairwise_correlation_20d",
    "correlation_dispersion_20d",
    "sector_correlation_to_market",
    "network_centrality_change",
]

ALL_FEATURES = NUMERIC_FEATURES + SENTIMENT_FEATURES + GRAPH_FEATURES


# Regime label mapping
REGIME_LABELS = {
    0: "LOW_RISK",
    1: "MODERATE_RISK",
    2: "HIGH_RISK",
    3: "EXTREME_RISK",
}

REGIME_TO_LABEL = {v: k for k, v in REGIME_LABELS.items()}
