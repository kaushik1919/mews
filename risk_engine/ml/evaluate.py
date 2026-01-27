"""
Evaluation metrics for ML risk models.

REQUIRED METRICS (from spec):
1. ROC-AUC (for classification)
2. Precision/recall during crisis periods
3. Calibration vs heuristic risk score
4. Lead-time analysis (early warning capability)

CRITICAL: ML models must be compared against Phase 4.0 heuristic,
not just against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_engine.ml.config import (
    ML_CONFIG,
    REGIME_LABELS,
    ModelType,
)
from risk_engine.ml.dataset import MLDataset
from risk_engine.ml.train import TrainedModel


@dataclass
class ClassificationMetrics:
    """Standard classification metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: float | None  # One-vs-rest ROC-AUC
    confusion_matrix: np.ndarray | None = None
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)


@dataclass
class CrisisPeriodMetrics:
    """Metrics for a specific crisis period."""

    period_name: str
    start_date: str
    end_date: str
    n_samples: int
    accuracy: float
    high_risk_recall: float  # Recall for HIGH_RISK + EXTREME_RISK
    avg_predicted_score: float
    avg_true_label: float


@dataclass
class LeadTimeAnalysis:
    """Analysis of early warning capability."""

    crisis_name: str
    crisis_start: str

    # For each lead window (5, 10, 20, 30 days):
    # Did the model elevate risk before crisis?
    lead_time_alerts: dict[int, bool]

    # Average risk score in lead window
    lead_time_scores: dict[int, float]


@dataclass
class HeuristicComparison:
    """Comparison against Phase 4.0 heuristic baseline."""

    ml_accuracy: float
    heuristic_accuracy: float
    accuracy_delta: float

    ml_crisis_recall: float
    heuristic_crisis_recall: float
    crisis_recall_delta: float

    correlation: float  # Correlation between ML and heuristic scores
    mean_absolute_diff: float  # Average absolute difference

    agreement_rate: float  # Fraction of samples with same regime


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a trained model.

    Includes all required metrics and comparisons.
    """

    model_type: ModelType
    split_name: str  # "val" or "test"
    n_samples: int

    classification: ClassificationMetrics
    crisis_periods: list[CrisisPeriodMetrics]
    lead_time: list[LeadTimeAnalysis]
    heuristic_comparison: HeuristicComparison | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "split": self.split_name,
            "n_samples": self.n_samples,
            "classification": {
                "accuracy": self.classification.accuracy,
                "precision_macro": self.classification.precision_macro,
                "recall_macro": self.classification.recall_macro,
                "f1_macro": self.classification.f1_macro,
                "roc_auc_ovr": self.classification.roc_auc_ovr,
                "per_class_precision": self.classification.per_class_precision,
                "per_class_recall": self.classification.per_class_recall,
            },
            "crisis_periods": [
                {
                    "name": cp.period_name,
                    "start": cp.start_date,
                    "end": cp.end_date,
                    "n_samples": cp.n_samples,
                    "accuracy": cp.accuracy,
                    "high_risk_recall": cp.high_risk_recall,
                }
                for cp in self.crisis_periods
            ],
            "lead_time": [
                {
                    "crisis": lt.crisis_name,
                    "alerts": lt.lead_time_alerts,
                    "scores": lt.lead_time_scores,
                }
                for lt in self.lead_time
            ],
            "heuristic_comparison": (
                {
                    "ml_accuracy": self.heuristic_comparison.ml_accuracy,
                    "heuristic_accuracy": self.heuristic_comparison.heuristic_accuracy,
                    "accuracy_delta": self.heuristic_comparison.accuracy_delta,
                    "agreement_rate": self.heuristic_comparison.agreement_rate,
                }
                if self.heuristic_comparison
                else None
            ),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)

    Returns:
        ClassificationMetrics
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precision_per = precision_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    recall_per = recall_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )

    per_class_precision = {
        REGIME_LABELS.get(int(c), str(c)): float(p)
        for c, p in zip(classes, precision_per, strict=False)
    }
    per_class_recall = {
        REGIME_LABELS.get(int(c), str(c)): float(r)
        for c, r in zip(classes, recall_per, strict=False)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC (one-vs-rest)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            # May fail if not all classes present in y_true
            pass

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
        roc_auc_ovr=roc_auc,
        confusion_matrix=cm,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
    )


def evaluate_crisis_period(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    period_start: str,
    period_end: str,
    period_name: str,
) -> CrisisPeriodMetrics | None:
    """
    Evaluate model performance during a specific crisis period.

    Args:
        timestamps: Sample timestamps
        y_true: True labels
        y_pred: Predicted labels
        y_score: Predicted risk scores [0, 1]
        period_start: Crisis start date
        period_end: Crisis end date
        period_name: Name of crisis period

    Returns:
        CrisisPeriodMetrics or None if no samples in period
    """
    # Filter to period
    mask = (timestamps >= period_start) & (timestamps <= period_end)
    n_samples = mask.sum()

    if n_samples == 0:
        return None

    y_true_period = y_true[mask]
    y_pred_period = y_pred[mask]
    y_score_period = y_score[mask]

    # Accuracy
    accuracy = float(np.mean(y_true_period == y_pred_period))

    # High-risk recall (labels 2 and 3)
    high_risk_mask = y_true_period >= 2
    if high_risk_mask.sum() > 0:
        high_risk_recall = float(
            np.mean(y_pred_period[high_risk_mask] >= 2)
        )
    else:
        high_risk_recall = 0.0

    return CrisisPeriodMetrics(
        period_name=period_name,
        start_date=period_start,
        end_date=period_end,
        n_samples=int(n_samples),
        accuracy=accuracy,
        high_risk_recall=high_risk_recall,
        avg_predicted_score=float(np.mean(y_score_period)),
        avg_true_label=float(np.mean(y_true_period)),
    )


def analyze_lead_time(
    timestamps: pd.DatetimeIndex,
    y_score: np.ndarray,
    crisis_start: str,
    crisis_name: str,
    lead_windows: list[int] | None = None,
    threshold: float = 0.5,
) -> LeadTimeAnalysis | None:
    """
    Analyze early warning capability before a crisis.

    Args:
        timestamps: Sample timestamps
        y_score: Predicted risk scores [0, 1]
        crisis_start: Crisis start date
        crisis_name: Name of crisis
        lead_windows: Days before crisis to check (default: [5, 10, 20, 30])
        threshold: Score threshold for "alert" (default: 0.5 = HIGH_RISK)

    Returns:
        LeadTimeAnalysis or None if crisis not in data
    """
    if lead_windows is None:
        lead_windows = ML_CONFIG.evaluation.lead_time_windows

    crisis_date = pd.Timestamp(crisis_start)

    # Check if crisis is in our data
    if crisis_date > timestamps.max() or crisis_date < timestamps.min():
        return None

    lead_time_alerts = {}
    lead_time_scores = {}

    for days in lead_windows:
        window_start = crisis_date - pd.Timedelta(days=days)
        window_end = crisis_date - pd.Timedelta(days=1)

        mask = (timestamps >= window_start) & (timestamps <= window_end)
        if mask.sum() == 0:
            lead_time_alerts[days] = False
            lead_time_scores[days] = 0.0
            continue

        scores_in_window = y_score[mask]
        avg_score = float(np.mean(scores_in_window))

        # Alert if average score exceeds threshold
        lead_time_alerts[days] = avg_score >= threshold
        lead_time_scores[days] = avg_score

    return LeadTimeAnalysis(
        crisis_name=crisis_name,
        crisis_start=crisis_start,
        lead_time_alerts=lead_time_alerts,
        lead_time_scores=lead_time_scores,
    )


def compare_to_heuristic(
    y_true: np.ndarray,
    ml_pred: np.ndarray,
    ml_score: np.ndarray,
    heuristic_pred: np.ndarray,
    heuristic_score: np.ndarray,
) -> HeuristicComparison:
    """
    Compare ML model to heuristic baseline.

    Args:
        y_true: True labels
        ml_pred: ML predicted labels
        ml_score: ML risk scores
        heuristic_pred: Heuristic predicted labels
        heuristic_score: Heuristic risk scores

    Returns:
        HeuristicComparison
    """
    ml_accuracy = float(np.mean(y_true == ml_pred))
    heuristic_accuracy = float(np.mean(y_true == heuristic_pred))

    # Crisis recall (HIGH_RISK + EXTREME_RISK)
    crisis_mask = y_true >= 2
    if crisis_mask.sum() > 0:
        ml_crisis_recall = float(np.mean(ml_pred[crisis_mask] >= 2))
        heuristic_crisis_recall = float(np.mean(heuristic_pred[crisis_mask] >= 2))
    else:
        ml_crisis_recall = 0.0
        heuristic_crisis_recall = 0.0

    # Score correlation
    correlation = float(np.corrcoef(ml_score, heuristic_score)[0, 1])

    # Mean absolute difference
    mean_abs_diff = float(np.mean(np.abs(ml_score - heuristic_score)))

    # Agreement rate
    agreement_rate = float(np.mean(ml_pred == heuristic_pred))

    return HeuristicComparison(
        ml_accuracy=ml_accuracy,
        heuristic_accuracy=heuristic_accuracy,
        accuracy_delta=ml_accuracy - heuristic_accuracy,
        ml_crisis_recall=ml_crisis_recall,
        heuristic_crisis_recall=heuristic_crisis_recall,
        crisis_recall_delta=ml_crisis_recall - heuristic_crisis_recall,
        correlation=correlation,
        mean_absolute_diff=mean_abs_diff,
        agreement_rate=agreement_rate,
    )


def evaluate_model(
    trained_model: TrainedModel,
    dataset: MLDataset,
    split: str = "test",
    heuristic_scores: np.ndarray | None = None,
    heuristic_preds: np.ndarray | None = None,
) -> EvaluationReport:
    """
    Complete evaluation of a trained model.

    Args:
        trained_model: Trained model to evaluate
        dataset: Dataset with test split
        split: Which split to evaluate ("val" or "test")
        heuristic_scores: Heuristic risk scores for comparison
        heuristic_preds: Heuristic regime predictions for comparison

    Returns:
        EvaluationReport with all metrics
    """
    # Get the right split
    if split == "test":
        data_split = dataset.test
    elif split == "val":
        data_split = dataset.val
    else:
        raise ValueError(f"Unknown split: {split}")

    # Get predictions
    y_pred = trained_model.predict(data_split.X)
    y_proba = trained_model.predict_proba(data_split.X)
    y_score = trained_model.predict_risk_score(data_split.X)

    # Classification metrics
    classification = compute_classification_metrics(
        data_split.y, y_pred, y_proba
    )

    # Crisis period metrics
    crisis_periods = []
    for start, end, name in ML_CONFIG.evaluation.crisis_periods:
        crisis_metric = evaluate_crisis_period(
            data_split.timestamps,
            data_split.y,
            y_pred,
            y_score,
            start,
            end,
            name,
        )
        if crisis_metric is not None:
            crisis_periods.append(crisis_metric)

    # Lead time analysis
    lead_time = []
    for start, _, name in ML_CONFIG.evaluation.crisis_periods:
        lt_analysis = analyze_lead_time(
            data_split.timestamps,
            y_score,
            start,
            name,
        )
        if lt_analysis is not None:
            lead_time.append(lt_analysis)

    # Heuristic comparison
    heuristic_comparison = None
    if heuristic_scores is not None and heuristic_preds is not None:
        heuristic_comparison = compare_to_heuristic(
            data_split.y,
            y_pred,
            y_score,
            heuristic_preds,
            heuristic_scores,
        )

    return EvaluationReport(
        model_type=trained_model.model_type,
        split_name=split,
        n_samples=data_split.n_samples,
        classification=classification,
        crisis_periods=crisis_periods,
        lead_time=lead_time,
        heuristic_comparison=heuristic_comparison,
    )


def print_evaluation_summary(report: EvaluationReport) -> None:
    """Print human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {report.model_type.value}")
    print(f"{'='*60}")
    print(f"Split: {report.split_name} ({report.n_samples} samples)")

    print("\nClassification Metrics:")
    print(f"  Accuracy:        {report.classification.accuracy:.4f}")
    print(f"  Precision (avg): {report.classification.precision_macro:.4f}")
    print(f"  Recall (avg):    {report.classification.recall_macro:.4f}")
    print(f"  F1 (avg):        {report.classification.f1_macro:.4f}")
    if report.classification.roc_auc_ovr:
        print(f"  ROC-AUC (OvR):   {report.classification.roc_auc_ovr:.4f}")

    if report.crisis_periods:
        print("\nCrisis Period Performance:")
        for cp in report.crisis_periods:
            print(f"  {cp.period_name}:")
            print(f"    Accuracy: {cp.accuracy:.4f}")
            print(f"    High-Risk Recall: {cp.high_risk_recall:.4f}")

    if report.lead_time:
        print("\nLead-Time Analysis:")
        for lt in report.lead_time:
            alerts = [f"{d}d" for d, a in lt.lead_time_alerts.items() if a]
            if alerts:
                print(f"  {lt.crisis_name}: Alert at {', '.join(alerts)} before")
            else:
                print(f"  {lt.crisis_name}: No early warning")

    if report.heuristic_comparison:
        hc = report.heuristic_comparison
        print("\nComparison vs Heuristic:")
        print(f"  ML Accuracy:        {hc.ml_accuracy:.4f}")
        print(f"  Heuristic Accuracy: {hc.heuristic_accuracy:.4f}")
        print(f"  Delta:              {hc.accuracy_delta:+.4f}")
        print(f"  Agreement Rate:     {hc.agreement_rate:.4f}")
        print(f"  Score Correlation:  {hc.correlation:.4f}")
