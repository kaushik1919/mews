"""
Risk Scoring Pipeline Step.

MEWS-FIN Phase 5A: Compute risk scores from features.

Calls Phase 4 risk engines and produces final ensemble score.
Mock mode generates plausible risk values.
"""

from __future__ import annotations

import numpy as np

from .context import PipelineContext


def run_risk(ctx: PipelineContext) -> None:
    """
    Execute risk scoring step.

    Computes heuristic, ML, and ensemble risk scores.
    In mock mode, generates synthetic scores.

    Args:
        ctx: Pipeline context (modified in place)
    """
    timing = ctx.start_step("risk")

    try:
        if ctx.config.mock_mode:
            _run_mock_risk(ctx)
        else:
            _run_live_risk(ctx)

        ctx.complete_step(timing, success=True)

    except Exception as e:
        ctx.complete_step(timing, success=False, error=str(e))
        raise


def _run_live_risk(ctx: PipelineContext) -> None:
    """
    Compute risk scores from actual features.

    Calls the Phase 4 risk engines.
    """
    features = ctx.features.all_features

    if not features:
        raise ValueError("No features available for risk scoring")

    # Compute heuristic score
    if ctx.config.risk.run_heuristic:
        _compute_heuristic_risk(ctx, features)

    # Compute ML scores
    if ctx.config.risk.run_ml:
        _compute_ml_risk(ctx, features)

    # Compute ensemble score
    if ctx.config.risk.run_ensemble:
        _compute_ensemble_risk(ctx)

    # Set final score
    _set_final_score(ctx)


def _compute_heuristic_risk(ctx: PipelineContext, features: dict) -> None:
    """Compute heuristic risk score."""
    try:
        from risk_engine.heuristic import compute_risk_score

        # Split features by type
        numeric = ctx.features.numeric_features
        sentiment = ctx.features.sentiment_features
        graph = ctx.features.graph_features

        snapshot = compute_risk_score(
            numeric_features=numeric,
            sentiment_features=sentiment,
            graph_features=graph,
            as_of=ctx.as_of.isoformat(),
        )

        ctx.risk.heuristic_score = snapshot.risk_score
        ctx.risk.sub_scores = snapshot.sub_scores

        # Extract dominant factors
        if snapshot.feature_contributions:
            sorted_contribs = sorted(
                snapshot.feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            ctx.risk.dominant_factors = sorted_contribs[:5]

        ctx.risk.regime_rationale = snapshot.regime_rationale

    except Exception as e:
        ctx.add_warning(f"Heuristic risk failed: {e}")
        _fill_mock_heuristic_risk(ctx)


def _compute_ml_risk(ctx: PipelineContext, features: dict) -> None:
    """Compute ML risk scores."""
    try:
        # ML inference would go here
        # For now, use mock since we don't have trained models in CI
        ctx.add_warning("ML inference not available - using mock scores")
        _fill_mock_ml_risk(ctx)

    except Exception as e:
        ctx.add_warning(f"ML risk failed: {e}")
        _fill_mock_ml_risk(ctx)


def _compute_ensemble_risk(ctx: PipelineContext) -> None:
    """Compute ensemble risk score."""
    try:
        from risk_engine.ensemble import compute_ensemble_risk_from_scores

        result = compute_ensemble_risk_from_scores(
            heuristic_score=ctx.risk.heuristic_score or 0.5,
            ml_scores=ctx.risk.ml_scores,
        )

        ctx.risk.ensemble_score = result.risk_score
        ctx.risk.regime = result.regime

        # Use ensemble explanation if available
        if result.explanation:
            if result.explanation.model_contributions:
                contrib_list = [
                    (name, value)
                    for name, value in result.explanation.model_contributions.items()
                ]
                # Merge with feature-level factors
                ctx.risk.dominant_factors = (
                    ctx.risk.dominant_factors[:3] if ctx.risk.dominant_factors else contrib_list[:3]
                )

    except Exception as e:
        ctx.add_warning(f"Ensemble risk failed: {e}")
        _fill_mock_ensemble_risk(ctx)


def _set_final_score(ctx: PipelineContext) -> None:
    """Set final score based on available scores."""
    # Prefer ensemble > heuristic > ML
    if ctx.risk.ensemble_score is not None:
        ctx.risk.final_score = ctx.risk.ensemble_score
    elif ctx.risk.heuristic_score is not None:
        ctx.risk.final_score = ctx.risk.heuristic_score
        ctx.risk.regime = _score_to_regime(ctx.risk.heuristic_score)
    elif ctx.risk.ml_scores:
        # Use primary ML model
        primary = ctx.config.risk.primary_ml_model
        if primary in ctx.risk.ml_scores:
            ctx.risk.final_score = ctx.risk.ml_scores[primary]
            ctx.risk.regime = _score_to_regime(ctx.risk.final_score)


def _score_to_regime(score: float) -> str:
    """Map score to regime."""
    if score >= 0.75:
        return "EXTREME_RISK"
    elif score >= 0.50:
        return "HIGH_RISK"
    elif score >= 0.25:
        return "MODERATE_RISK"
    else:
        return "LOW_RISK"


def _run_mock_risk(ctx: PipelineContext) -> None:
    """Generate mock risk scores for testing."""
    _fill_mock_heuristic_risk(ctx)
    _fill_mock_ml_risk(ctx)
    _fill_mock_ensemble_risk(ctx)
    _set_final_score(ctx)
    ctx.risk.metadata["mode"] = "mock"


def _fill_mock_heuristic_risk(ctx: PipelineContext) -> None:
    """Generate mock heuristic risk score."""
    np.random.seed(49)

    ctx.risk.heuristic_score = np.random.uniform(0.45, 0.72)

    ctx.risk.sub_scores = {
        "volatility_risk": np.random.uniform(0.50, 0.80),
        "correlation_risk": np.random.uniform(0.45, 0.75),
        "liquidity_risk": np.random.uniform(0.35, 0.65),
        "sentiment_risk": np.random.uniform(0.40, 0.70),
        "credit_risk": np.random.uniform(0.40, 0.65),
    }

    ctx.risk.dominant_factors = [
        ("realized_volatility_20d", np.random.uniform(0.12, 0.22)),
        ("avg_pairwise_correlation_20d", np.random.uniform(0.10, 0.18)),
        ("news_sentiment_5d", np.random.uniform(0.08, 0.15)),
    ]

    ctx.risk.regime_rationale = (
        "Elevated volatility and rising cross-asset correlations indicate "
        "systemic stress consistent with risk-off regimes. Sentiment remains "
        "cautious but not panicked."
    )


def _fill_mock_ml_risk(ctx: PipelineContext) -> None:
    """Generate mock ML risk scores."""
    np.random.seed(50)

    ctx.risk.ml_scores = {
        "random_forest": np.random.uniform(0.50, 0.75),
    }

    # Only add xgboost sometimes for realism
    if np.random.random() > 0.3:
        ctx.risk.ml_scores["xgboost"] = np.random.uniform(0.48, 0.73)


def _fill_mock_ensemble_risk(ctx: PipelineContext) -> None:
    """Generate mock ensemble risk score."""
    np.random.seed(51)

    # Ensemble should be weighted average of heuristic and ML
    h_score = ctx.risk.heuristic_score or 0.55
    ml_score = ctx.risk.ml_scores.get("random_forest", 0.60)

    # Ensemble: 35% heuristic, 65% ML
    ensemble = 0.35 * h_score + 0.65 * ml_score

    # Add small calibration adjustment
    ensemble = ensemble + np.random.uniform(-0.03, 0.03)
    ensemble = max(0.0, min(1.0, ensemble))

    ctx.risk.ensemble_score = ensemble
    ctx.risk.regime = _score_to_regime(ensemble)
