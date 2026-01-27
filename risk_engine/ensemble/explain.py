"""
Ensemble Explainability Module.

MEWS-FIN Phase 4.2: Generate human-readable explanations for
ensemble risk scores, combining insights from heuristic and ML models.

Key Requirements (MANDATORY):
    - Show contribution from each model (heuristic + ML)
    - Show top features (from SHAP + heuristic)
    - Explain why ensemble differs from individual models
    - Provide regime rationale

Explainability > Marginal Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelContribution:
    """Contribution of a single model to ensemble score."""

    model_name: str
    raw_score: float  # Model's original score
    weight: float  # Weight in ensemble
    weighted_contribution: float  # weight * raw_score
    contribution_pct: float  # Percentage of final ensemble score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "raw_score": self.raw_score,
            "weight": self.weight,
            "weighted_contribution": self.weighted_contribution,
            "contribution_pct": self.contribution_pct,
        }


@dataclass
class FeatureContribution:
    """Contribution of a feature across models."""

    feature_name: str
    heuristic_contribution: float | None  # From heuristic sub-scores
    ml_shap_value: float | None  # From SHAP
    combined_importance: float  # Weighted importance
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "heuristic_contribution": self.heuristic_contribution,
            "ml_shap_value": self.ml_shap_value,
            "combined_importance": self.combined_importance,
            "rank": self.rank,
        }


@dataclass
class EnsembleDelta:
    """Explanation of difference between ensemble and component models."""

    ensemble_vs_heuristic: float  # ensemble - heuristic
    ensemble_vs_primary_ml: float  # ensemble - primary ML
    heuristic_vs_ml: float  # heuristic - ML
    dominant_direction: str  # "ml_higher", "heuristic_higher", "aligned"
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_vs_heuristic": self.ensemble_vs_heuristic,
            "ensemble_vs_primary_ml": self.ensemble_vs_primary_ml,
            "heuristic_vs_ml": self.heuristic_vs_ml,
            "dominant_direction": self.dominant_direction,
            "explanation": self.explanation,
        }


@dataclass
class EnsembleExplanation:
    """
    Complete explanation for an ensemble risk score.

    This is the primary output for explainability.
    """

    # Model contributions
    model_contributions: list[ModelContribution]
    dominant_model: str  # Model with highest contribution
    agreement_level: str  # "strong", "moderate", "weak"

    # Feature contributions (merged from heuristic + SHAP)
    top_features: list[FeatureContribution]

    # Delta analysis
    delta_analysis: EnsembleDelta

    # Regime explanation
    regime: str
    regime_rationale: str

    # Summary
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_contributions": [mc.to_dict() for mc in self.model_contributions],
            "dominant_model": self.dominant_model,
            "agreement_level": self.agreement_level,
            "top_features": [fc.to_dict() for fc in self.top_features],
            "delta_analysis": self.delta_analysis.to_dict(),
            "regime": self.regime,
            "regime_rationale": self.regime_rationale,
            "summary": self.summary,
        }


def compute_ensemble_explanation(
    ensemble_score: float,
    heuristic_score: float,
    heuristic_contributions: dict[str, float],
    ml_scores: dict[str, float],
    ml_shap_values: dict[str, float] | None,
    model_weights: dict[str, float],
    regime: str,
    n_top_features: int = 5,
) -> EnsembleExplanation:
    """
    Compute comprehensive explanation for ensemble score.

    Args:
        ensemble_score: Final ensemble risk score
        heuristic_score: Heuristic model score
        heuristic_contributions: Feature contributions from heuristic
        ml_scores: Scores from each ML model
        ml_shap_values: SHAP values from primary ML model (optional)
        model_weights: Weights used for each model
        regime: Assigned risk regime
        n_top_features: Number of top features to include

    Returns:
        Complete ensemble explanation
    """
    # Compute model contributions
    model_contribs = _compute_model_contributions(
        heuristic_score, ml_scores, model_weights, ensemble_score
    )

    # Find dominant model
    dominant_model = max(model_contribs, key=lambda mc: mc.contribution_pct).model_name

    # Compute agreement level
    agreement = _compute_agreement_level(heuristic_score, ml_scores)

    # Merge feature contributions
    top_features = _merge_feature_contributions(
        heuristic_contributions, ml_shap_values, model_weights, n_top_features
    )

    # Delta analysis
    primary_ml_score = list(ml_scores.values())[0] if ml_scores else heuristic_score
    delta = _compute_delta_analysis(ensemble_score, heuristic_score, primary_ml_score)

    # Regime rationale
    regime_rationale = _generate_regime_rationale(
        regime, ensemble_score, top_features, model_contribs
    )

    # Summary
    summary = _generate_summary(
        ensemble_score, regime, dominant_model, agreement, top_features
    )

    return EnsembleExplanation(
        model_contributions=model_contribs,
        dominant_model=dominant_model,
        agreement_level=agreement,
        top_features=top_features,
        delta_analysis=delta,
        regime=regime,
        regime_rationale=regime_rationale,
        summary=summary,
    )


def _compute_model_contributions(
    heuristic_score: float,
    ml_scores: dict[str, float],
    weights: dict[str, float],
    ensemble_score: float,
) -> list[ModelContribution]:
    """Compute contribution breakdown by model."""
    contributions = []

    # Heuristic contribution
    h_weight = weights.get("heuristic", 0.35)
    h_contrib = h_weight * heuristic_score
    contributions.append(ModelContribution(
        model_name="heuristic",
        raw_score=heuristic_score,
        weight=h_weight,
        weighted_contribution=h_contrib,
        contribution_pct=h_contrib / ensemble_score if ensemble_score > 0 else 0,
    ))

    # ML model contributions
    for model_name, score in ml_scores.items():
        weight = weights.get(model_name, 0)
        if weight > 0:
            contrib = weight * score
            contributions.append(ModelContribution(
                model_name=model_name,
                raw_score=score,
                weight=weight,
                weighted_contribution=contrib,
                contribution_pct=contrib / ensemble_score if ensemble_score > 0 else 0,
            ))

    return contributions


def _compute_agreement_level(
    heuristic_score: float,
    ml_scores: dict[str, float],
) -> str:
    """
    Compute how well models agree.

    Agreement levels:
    - strong: all models within 0.10 of each other
    - moderate: all models within 0.25 of each other
    - weak: models differ by more than 0.25
    """
    all_scores = [heuristic_score] + list(ml_scores.values())

    if len(all_scores) < 2:
        return "strong"

    max_diff = max(all_scores) - min(all_scores)

    if max_diff <= 0.10:
        return "strong"
    elif max_diff <= 0.25:
        return "moderate"
    else:
        return "weak"


def _merge_feature_contributions(
    heuristic_contributions: dict[str, float],
    ml_shap_values: dict[str, float] | None,
    weights: dict[str, float],
    n_top: int,
) -> list[FeatureContribution]:
    """
    Merge feature importance from heuristic and ML models.

    Uses weighted combination based on model weights.
    """
    h_weight = weights.get("heuristic", 0.35)
    ml_weight = 1.0 - h_weight

    # Collect all features
    all_features: set[str] = set(heuristic_contributions.keys())
    if ml_shap_values:
        all_features.update(ml_shap_values.keys())

    # Compute combined importance
    combined: list[tuple[str, float, float | None, float | None]] = []

    for feature in all_features:
        h_contrib = heuristic_contributions.get(feature)
        ml_shap = ml_shap_values.get(feature) if ml_shap_values else None

        # Compute combined importance
        h_val = abs(h_contrib) if h_contrib is not None else 0.0
        ml_val = abs(ml_shap) if ml_shap is not None else 0.0

        combined_imp = h_weight * h_val + ml_weight * ml_val
        combined.append((feature, combined_imp, h_contrib, ml_shap))

    # Sort by combined importance
    combined.sort(key=lambda x: x[1], reverse=True)

    # Create top-N feature contributions
    top_features = []
    for rank, (feature, importance, h_contrib, ml_shap) in enumerate(combined[:n_top], 1):
        top_features.append(FeatureContribution(
            feature_name=feature,
            heuristic_contribution=h_contrib,
            ml_shap_value=ml_shap,
            combined_importance=importance,
            rank=rank,
        ))

    return top_features


def _compute_delta_analysis(
    ensemble_score: float,
    heuristic_score: float,
    primary_ml_score: float,
) -> EnsembleDelta:
    """Analyze differences between ensemble and component models."""
    delta_h = ensemble_score - heuristic_score
    delta_ml = ensemble_score - primary_ml_score
    h_vs_ml = heuristic_score - primary_ml_score

    # Determine dominant direction
    if abs(h_vs_ml) < 0.05:
        direction = "aligned"
        explanation = (
            f"Heuristic and ML models are closely aligned "
            f"(difference: {abs(h_vs_ml):.3f}). Ensemble reflects consensus."
        )
    elif h_vs_ml > 0:
        direction = "heuristic_higher"
        explanation = (
            f"Heuristic indicates higher risk ({heuristic_score:.3f}) than ML "
            f"({primary_ml_score:.3f}). Ensemble balances at {ensemble_score:.3f}. "
            f"Heuristic may be detecting structural risks not in training data."
        )
    else:
        direction = "ml_higher"
        explanation = (
            f"ML model indicates higher risk ({primary_ml_score:.3f}) than heuristic "
            f"({heuristic_score:.3f}). Ensemble balances at {ensemble_score:.3f}. "
            f"ML may be detecting pattern-based risks."
        )

    return EnsembleDelta(
        ensemble_vs_heuristic=delta_h,
        ensemble_vs_primary_ml=delta_ml,
        heuristic_vs_ml=h_vs_ml,
        dominant_direction=direction,
        explanation=explanation,
    )


def _generate_regime_rationale(
    regime: str,
    score: float,
    top_features: list[FeatureContribution],
    model_contribs: list[ModelContribution],
) -> str:
    """Generate human-readable regime explanation."""
    # Get dominant features
    feature_names = [f.feature_name for f in top_features[:3]]
    features_str = ", ".join(feature_names) if feature_names else "multiple factors"

    # Get dominant model
    dominant = max(model_contribs, key=lambda mc: mc.contribution_pct)

    if regime == "LOW_RISK":
        return (
            f"Risk score of {score:.3f} indicates stable market conditions. "
            f"Key indicators ({features_str}) are within normal ranges. "
            f"Both heuristic and ML models agree on low stress levels."
        )
    elif regime == "MODERATE_RISK":
        return (
            f"Risk score of {score:.3f} signals elevated attention warranted. "
            f"Driven primarily by {features_str}. "
            f"{dominant.model_name.title()} model contributes {dominant.contribution_pct:.1%} of signal."
        )
    elif regime == "HIGH_RISK":
        return (
            f"Risk score of {score:.3f} indicates significant market stress. "
            f"Multiple indicators elevated: {features_str}. "
            f"Defensive posture recommended. "
            f"{dominant.model_name.title()} model is primary contributor."
        )
    else:  # EXTREME_RISK
        return (
            f"Risk score of {score:.3f} signals extreme stress conditions. "
            f"Critical drivers: {features_str}. "
            f"Immediate attention required. "
            f"All models indicate elevated risk levels."
        )


def _generate_summary(
    score: float,
    regime: str,
    dominant_model: str,
    agreement: str,
    top_features: list[FeatureContribution],
) -> str:
    """Generate one-line summary."""
    top_feature = top_features[0].feature_name if top_features else "unknown"

    agreement_text = {
        "strong": "Models strongly agree",
        "moderate": "Models moderately agree",
        "weak": "Models show divergence",
    }[agreement]

    return (
        f"Ensemble risk {score:.3f} ({regime}). "
        f"{agreement_text}. "
        f"Top driver: {top_feature}. "
        f"Primary signal from {dominant_model}."
    )
