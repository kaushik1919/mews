"""
SHAP explainability for ML risk models.

CRITICAL REQUIREMENT: Explainability is not optional.

For each model, we must:
1. Compute SHAP values
2. Produce global feature importance
3. Produce per-period explanations
4. Compare SHAP explanations vs heuristic feature contributions

SHAP provides model-agnostic explanations that are:
- Theoretically grounded (Shapley values)
- Consistent across model types
- Comparable to heuristic feature contributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_engine.ml.config import ML_CONFIG, ModelType
from risk_engine.ml.dataset import DatasetSplit
from risk_engine.ml.train import TrainedModel


@dataclass
class FeatureImportance:
    """Global feature importance from SHAP."""

    feature_name: str
    mean_abs_shap: float  # Mean absolute SHAP value
    mean_shap: float  # Mean SHAP value (signed)
    std_shap: float  # Standard deviation


@dataclass
class SampleExplanation:
    """SHAP explanation for a single sample."""

    timestamp: pd.Timestamp
    predicted_label: int
    predicted_score: float
    true_label: int
    shap_values: dict[str, float]  # Feature -> SHAP value
    top_contributors: list[str]  # Top 5 features by |SHAP|


@dataclass
class PeriodExplanation:
    """SHAP explanation aggregated over a time period."""

    period_name: str
    start_date: str
    end_date: str
    n_samples: int
    avg_predicted_score: float
    feature_importance: list[FeatureImportance]
    top_features: list[str]


@dataclass
class HeuristicComparison:
    """Comparison of SHAP explanations to heuristic contributions."""

    feature_name: str
    shap_importance: float  # From ML model
    heuristic_weight: float  # From heuristic
    rank_difference: int  # Difference in importance ranking


@dataclass
class SHAPExplanation:
    """
    Complete SHAP explanation for a model.

    Contains global importance, local explanations, and
    comparison to heuristic baseline.
    """

    model_type: ModelType
    n_samples_explained: int

    # Global importance
    global_importance: list[FeatureImportance]

    # Per-sample explanations (subset)
    sample_explanations: list[SampleExplanation]

    # Per-period explanations
    period_explanations: list[PeriodExplanation]

    # Comparison to heuristic
    heuristic_comparison: list[HeuristicComparison]

    # Raw SHAP values for further analysis
    shap_values: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "n_samples": self.n_samples_explained,
            "global_importance": [
                {
                    "feature": fi.feature_name,
                    "mean_abs_shap": fi.mean_abs_shap,
                    "mean_shap": fi.mean_shap,
                }
                for fi in self.global_importance
            ],
            "top_features": [fi.feature_name for fi in self.global_importance[:5]],
            "period_explanations": [
                {
                    "period": pe.period_name,
                    "top_features": pe.top_features,
                    "avg_score": pe.avg_predicted_score,
                }
                for pe in self.period_explanations
            ],
        }

    def get_summary_text(self) -> str:
        """Generate textual summary for reporting."""
        lines = [
            f"SHAP Explanation Summary: {self.model_type.value}",
            f"Samples analyzed: {self.n_samples_explained}",
            "",
            "Top 5 Features by Importance:",
        ]

        for i, fi in enumerate(self.global_importance[:5], 1):
            lines.append(
                f"  {i}. {fi.feature_name}: "
                f"mean|SHAP|={fi.mean_abs_shap:.4f}"
            )

        if self.period_explanations:
            lines.append("")
            lines.append("Period Analysis:")
            for pe in self.period_explanations[:3]:
                lines.append(
                    f"  {pe.period_name}: top features = "
                    f"{', '.join(pe.top_features[:3])}"
                )

        return "\n".join(lines)


def create_shap_explainer(
    trained_model: TrainedModel,
    background_data: np.ndarray,
) -> Any:
    """
    Create appropriate SHAP explainer for model type.

    Uses TreeExplainer for tree models, KernelExplainer for linear.

    Args:
        trained_model: Trained model
        background_data: Background samples for explainer

    Returns:
        SHAP explainer instance
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap") from None

    model = trained_model.model.model  # Get underlying sklearn/xgb model

    if trained_model.model_type in [
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
    ]:
        # TreeExplainer for tree-based models
        return shap.TreeExplainer(model)
    else:
        # KernelExplainer for linear models
        # Use a subset of background data for efficiency
        n_background = min(len(background_data), ML_CONFIG.shap.background_samples)
        background_subset = background_data[:n_background]

        def predict_fn(X):
            return trained_model.predict_proba(X)

        return shap.KernelExplainer(predict_fn, background_subset)


def compute_shap_values(
    trained_model: TrainedModel,
    X: np.ndarray,
    background_data: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute SHAP values for given samples.

    Args:
        trained_model: Trained model
        X: Samples to explain
        background_data: Background data for explainer

    Returns:
        SHAP values array (n_samples, n_features) or
        (n_samples, n_features, n_classes) for multi-class
    """
    import importlib.util
    if importlib.util.find_spec("shap") is None:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    # Limit samples for performance
    max_samples = ML_CONFIG.shap.max_samples
    if len(X) > max_samples:
        X = X[:max_samples]

    if background_data is None:
        background_data = X

    explainer = create_shap_explainer(trained_model, background_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    # Handle different return formats
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, one per class
        # Take mean across classes for interpretability
        shap_values = np.mean(np.array(shap_values), axis=0)

    return shap_values


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """
    Compute global feature importance from SHAP values.

    Args:
        shap_values: SHAP values (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        List of FeatureImportance, sorted by importance
    """
    importances = []

    for i, name in enumerate(feature_names):
        feature_shap = shap_values[:, i]
        importances.append(
            FeatureImportance(
                feature_name=name,
                mean_abs_shap=float(np.mean(np.abs(feature_shap))),
                mean_shap=float(np.mean(feature_shap)),
                std_shap=float(np.std(feature_shap)),
            )
        )

    # Sort by mean absolute SHAP
    importances.sort(key=lambda x: x.mean_abs_shap, reverse=True)
    return importances


def compute_sample_explanations(
    shap_values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    y_true: np.ndarray,
    feature_names: list[str],
    n_samples: int = 10,
) -> list[SampleExplanation]:
    """
    Compute explanations for individual samples.

    Args:
        shap_values: SHAP values
        timestamps: Sample timestamps
        y_pred: Predicted labels
        y_score: Predicted scores
        y_true: True labels
        feature_names: Feature names
        n_samples: Number of samples to include

    Returns:
        List of SampleExplanation
    """
    explanations = []
    n = min(n_samples, len(shap_values))

    # Select samples evenly spread across the dataset
    indices = np.linspace(0, len(shap_values) - 1, n, dtype=int)

    for idx in indices:
        shap_dict = {
            name: float(shap_values[idx, i])
            for i, name in enumerate(feature_names)
        }

        # Top contributors by absolute SHAP
        sorted_features = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_contributors = [name for name, _ in sorted_features[:5]]

        explanations.append(
            SampleExplanation(
                timestamp=timestamps[idx],
                predicted_label=int(y_pred[idx]),
                predicted_score=float(y_score[idx]),
                true_label=int(y_true[idx]),
                shap_values=shap_dict,
                top_contributors=top_contributors,
            )
        )

    return explanations


def compute_period_explanations(
    shap_values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    y_score: np.ndarray,
    feature_names: list[str],
    periods: list[tuple[str, str, str]] | None = None,
) -> list[PeriodExplanation]:
    """
    Compute aggregated explanations for specific periods.

    Args:
        shap_values: SHAP values
        timestamps: Sample timestamps
        y_score: Predicted scores
        feature_names: Feature names
        periods: List of (start, end, name) tuples

    Returns:
        List of PeriodExplanation
    """
    if periods is None:
        periods = ML_CONFIG.evaluation.crisis_periods

    explanations = []

    for start, end, name in periods:
        mask = (timestamps >= start) & (timestamps <= end)
        n_in_period = mask.sum()

        if n_in_period == 0:
            continue

        period_shap = shap_values[mask]
        period_scores = y_score[mask]

        # Compute importance for this period
        importance = compute_global_importance(period_shap, feature_names)

        explanations.append(
            PeriodExplanation(
                period_name=name,
                start_date=start,
                end_date=end,
                n_samples=int(n_in_period),
                avg_predicted_score=float(np.mean(period_scores)),
                feature_importance=importance,
                top_features=[fi.feature_name for fi in importance[:5]],
            )
        )

    return explanations


def compare_to_heuristic_weights(
    global_importance: list[FeatureImportance],
    heuristic_weights: dict[str, float] | None = None,
) -> list[HeuristicComparison]:
    """
    Compare SHAP importance to heuristic feature weights.

    Args:
        global_importance: SHAP global importance
        heuristic_weights: Heuristic feature weights (optional)

    Returns:
        List of HeuristicComparison
    """
    # Default heuristic weights from Phase 4.0
    if heuristic_weights is None:
        from risk_engine.heuristic.weights import SUB_SCORE_DEFINITIONS

        heuristic_weights = {}
        for sub_def in SUB_SCORE_DEFINITIONS.values():
            for feature, weight in sub_def.feature_weights.items():
                # Normalize by sub-score weight
                heuristic_weights[feature] = weight

    comparisons = []

    # Create SHAP ranking
    shap_ranking = {
        fi.feature_name: (i, fi.mean_abs_shap)
        for i, fi in enumerate(global_importance)
    }

    # Create heuristic ranking
    heuristic_sorted = sorted(
        heuristic_weights.items(), key=lambda x: x[1], reverse=True
    )
    heuristic_ranking = {name: i for i, (name, _) in enumerate(heuristic_sorted)}

    for feature, weight in heuristic_weights.items():
        if feature in shap_ranking:
            shap_rank, shap_importance = shap_ranking[feature]
            heur_rank = heuristic_ranking[feature]

            comparisons.append(
                HeuristicComparison(
                    feature_name=feature,
                    shap_importance=shap_importance,
                    heuristic_weight=weight,
                    rank_difference=shap_rank - heur_rank,
                )
            )

    # Sort by absolute rank difference
    comparisons.sort(key=lambda x: abs(x.rank_difference))
    return comparisons


def compute_shap_explanations(
    trained_model: TrainedModel,
    data_split: DatasetSplit,
    background_data: np.ndarray | None = None,
    compute_local: bool | None = None,
) -> SHAPExplanation:
    """
    Compute complete SHAP explanation for a model.

    Args:
        trained_model: Trained model to explain
        data_split: Data split to explain
        background_data: Background data for explainer
        compute_local: Whether to compute per-sample explanations

    Returns:
        SHAPExplanation with all explanations
    """
    if compute_local is None:
        compute_local = ML_CONFIG.shap.compute_local_explanations

    if background_data is None:
        background_data = data_split.X

    # Compute SHAP values
    shap_values = compute_shap_values(
        trained_model, data_split.X, background_data
    )

    # Limit samples if SHAP values were truncated
    n_explained = len(shap_values)

    # Get predictions
    y_pred = trained_model.predict(data_split.X[:n_explained])
    y_score = trained_model.predict_risk_score(data_split.X[:n_explained])

    # Global importance
    global_importance = compute_global_importance(
        shap_values, trained_model.feature_names
    )

    # Sample explanations
    if compute_local:
        sample_explanations = compute_sample_explanations(
            shap_values,
            data_split.timestamps[:n_explained],
            y_pred,
            y_score,
            data_split.y[:n_explained],
            trained_model.feature_names,
        )
    else:
        sample_explanations = []

    # Period explanations
    period_explanations = compute_period_explanations(
        shap_values,
        data_split.timestamps[:n_explained],
        y_score,
        trained_model.feature_names,
    )

    # Heuristic comparison
    heuristic_comparison = compare_to_heuristic_weights(global_importance)

    return SHAPExplanation(
        model_type=trained_model.model_type,
        n_samples_explained=n_explained,
        global_importance=global_importance,
        sample_explanations=sample_explanations,
        period_explanations=period_explanations,
        heuristic_comparison=heuristic_comparison,
        shap_values=shap_values,
        feature_names=trained_model.feature_names,
    )


def print_shap_summary(explanation: SHAPExplanation) -> None:
    """Print human-readable SHAP summary."""
    print(explanation.get_summary_text())

    if explanation.heuristic_comparison:
        print("\nComparison to Heuristic:")
        for hc in explanation.heuristic_comparison[:5]:
            direction = "↑" if hc.rank_difference < 0 else "↓" if hc.rank_difference > 0 else "="
            print(
                f"  {hc.feature_name}: "
                f"SHAP={hc.shap_importance:.4f}, "
                f"Heur={hc.heuristic_weight:.2f} "
                f"({direction} rank diff: {hc.rank_difference})"
            )
