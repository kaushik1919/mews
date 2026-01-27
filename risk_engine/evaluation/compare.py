"""
Model Comparison for MEWS-FIN Evaluation.

Phase 4.3: Compare heuristic vs ML vs ensemble performance.

Key Questions:
    - Which approach provides the best lead time?
    - Which has the lowest false-positive burden?
    - Does ensemble outperform individual approaches?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .crises import CrisisWindow, get_crisis_windows
from .false_positives import (
    FalsePositiveResult,
    compute_false_positives,
    false_positive_burden_score,
)
from .lead_time import (
    EVALUATION_THRESHOLDS,
    LeadTimeSummary,
    compute_lead_time_summary,
)


@dataclass
class ModelEvaluation:
    """
    Complete evaluation for a single model.

    Attributes:
        model_name: Name of the model (heuristic, ml, ensemble)
        lead_times: Lead-time summary per threshold
        false_positives: False-positive result per threshold
        best_threshold: Threshold with best trade-off
        overall_score: Composite performance score (higher is better)
    """

    model_name: str
    lead_times: dict[float, LeadTimeSummary]
    false_positives: dict[float, FalsePositiveResult]
    best_threshold: float
    overall_score: float


@dataclass
class ModelComparison:
    """
    Comparison across all models.

    Attributes:
        models: Individual model evaluations
        best_model: Name of best performing model
        recommended_threshold: Best threshold across all models
        summary: Human-readable summary
    """

    models: dict[str, ModelEvaluation]
    best_model: str
    recommended_threshold: float
    summary: str


def compute_model_score(
    lead_time: LeadTimeSummary,
    false_positive: FalsePositiveResult,
    lead_time_weight: float = 0.6,
    fp_weight: float = 0.4,
    target_lead_time: int = 30,
) -> float:
    """
    Compute composite model score.

    Higher is better. Balances lead time against false positive burden.

    Args:
        lead_time: Lead time summary
        false_positive: False positive result
        lead_time_weight: Weight for lead time component
        fp_weight: Weight for false positive component
        target_lead_time: Target lead time for normalization

    Returns:
        Score in [0, 1] where 1 is perfect
    """
    # Lead time component: fraction of target achieved (capped at 1)
    if lead_time.median_lead_time is not None:
        lead_score = min(lead_time.median_lead_time / target_lead_time, 1.0)
    else:
        lead_score = 0.0  # Missed all crises

    # Detection rate component
    detection_score = lead_time.detection_rate

    # Combined lead time score
    combined_lead = 0.5 * lead_score + 0.5 * detection_score

    # False positive burden (invert: lower burden = higher score)
    fp_burden = false_positive_burden_score(false_positive)
    fp_score = max(0.0, 1.0 - fp_burden)

    return lead_time_weight * combined_lead + fp_weight * fp_score


def evaluate_model(
    risk_scores: dict[date, float],
    model_name: str,
    thresholds: Sequence[float] | None = None,
    crises: Sequence[CrisisWindow] | None = None,
) -> ModelEvaluation:
    """
    Evaluate a single model across all thresholds.

    Args:
        risk_scores: Dict of date -> risk_score
        model_name: Name for this model
        thresholds: Thresholds to evaluate
        crises: Crisis windows

    Returns:
        ModelEvaluation with complete metrics
    """
    if thresholds is None:
        thresholds = EVALUATION_THRESHOLDS
    if crises is None:
        crises = get_crisis_windows()

    lead_times: dict[float, LeadTimeSummary] = {}
    false_positives: dict[float, FalsePositiveResult] = {}
    scores: dict[float, float] = {}

    for threshold in thresholds:
        lt = compute_lead_time_summary(risk_scores, threshold, crises)
        fp = compute_false_positives(risk_scores, threshold, crises)
        lead_times[threshold] = lt
        false_positives[threshold] = fp
        scores[threshold] = compute_model_score(lt, fp)

    # Find best threshold
    best_threshold = max(scores, key=lambda t: scores[t])
    overall_score = scores[best_threshold]

    return ModelEvaluation(
        model_name=model_name,
        lead_times=lead_times,
        false_positives=false_positives,
        best_threshold=best_threshold,
        overall_score=overall_score,
    )


def compare_models(
    heuristic_scores: dict[date, float],
    ml_scores: dict[date, float],
    ensemble_scores: dict[date, float],
    thresholds: Sequence[float] | None = None,
    crises: Sequence[CrisisWindow] | None = None,
) -> ModelComparison:
    """
    Compare heuristic, ML, and ensemble models.

    Args:
        heuristic_scores: Heuristic model risk scores by date
        ml_scores: Best ML model risk scores by date
        ensemble_scores: Ensemble risk scores by date
        thresholds: Thresholds to evaluate
        crises: Crisis windows

    Returns:
        ModelComparison with rankings and recommendations
    """
    models: dict[str, ModelEvaluation] = {}

    models["heuristic"] = evaluate_model(
        heuristic_scores, "Heuristic (Phase 4.0)", thresholds, crises
    )
    models["ml"] = evaluate_model(
        ml_scores, "Best ML (Phase 4.1)", thresholds, crises
    )
    models["ensemble"] = evaluate_model(
        ensemble_scores, "Ensemble (Phase 4.2)", thresholds, crises
    )

    # Find best model
    best_model = max(models, key=lambda m: models[m].overall_score)
    recommended_threshold = models[best_model].best_threshold

    # Generate summary
    best_eval = models[best_model]
    best_lt = best_eval.lead_times[recommended_threshold]
    best_fp = best_eval.false_positives[recommended_threshold]

    median_lt = best_lt.median_lead_time if best_lt.median_lead_time else 0
    fp_rate = best_fp.false_positive_rate * 100

    summary = (
        f"The MEWS-FIN {best_eval.model_name} provides {median_lt:.0f} days median "
        f"early warning with a {fp_rate:.1f}% false-positive rate at threshold "
        f"{recommended_threshold:.2f}."
    )

    # Add comparison details
    heuristic_score = models["heuristic"].overall_score
    ml_score = models["ml"].overall_score
    ensemble_score = models["ensemble"].overall_score

    if best_model == "ensemble":
        improvement_h = ((ensemble_score / heuristic_score) - 1) * 100 if heuristic_score > 0 else 0
        improvement_m = ((ensemble_score / ml_score) - 1) * 100 if ml_score > 0 else 0
        summary += (
            f" This outperforms the heuristic baseline by {improvement_h:.1f}% "
            f"and the ML baseline by {improvement_m:.1f}%."
        )

    return ModelComparison(
        models=models,
        best_model=best_model,
        recommended_threshold=recommended_threshold,
        summary=summary,
    )


def quick_compare(
    heuristic_scores: dict[date, float],
    ml_scores: dict[date, float],
    ensemble_scores: dict[date, float],
    threshold: float = 0.60,
) -> dict[str, dict[str, float | int | None]]:
    """
    Quick comparison at a single threshold.

    Useful for fast evaluation during development.

    Returns:
        Dict with model name -> metrics
    """
    crises = get_crisis_windows()

    results: dict[str, dict[str, float | int | None]] = {}

    for name, scores in [
        ("heuristic", heuristic_scores),
        ("ml", ml_scores),
        ("ensemble", ensemble_scores),
    ]:
        lt = compute_lead_time_summary(scores, threshold, crises)
        fp = compute_false_positives(scores, threshold, crises)

        results[name] = {
            "median_lead_time": lt.median_lead_time,
            "detection_rate": lt.detection_rate,
            "false_positive_rate": fp.false_positive_rate,
            "max_streak": fp.max_streak_duration,
            "score": compute_model_score(lt, fp),
        }

    return results
