"""
Heuristic Risk Engine for MEWS-FIN.

Phase 4.0: Baseline Explainable Risk Scoring

This module computes a single interpretable risk score by combining
multimodal feature snapshots using explicit rules and weights.

NOT a prediction model. NOT a trading signal.
This is a baseline for:
- Validating feature pipelines
- Comparing against ML models
- Debugging explainability
- Demonstrating end-to-end flow

Features:
- Deterministic (same inputs → same output)
- Explainable (feature contributions always available)
- Weight-based (no learned parameters)
- Replaceable (interface supports ML swap-in)

Sub-scores:
- volatility_risk: Realized and implied volatility signals
- correlation_risk: Cross-asset correlation dynamics
- liquidity_risk: Volume and market functioning
- sentiment_risk: News sentiment deterioration
- credit_risk: Macro/credit indicators (null-safe)

Public API:
    from risk_engine.heuristic import compute_risk_score, RiskScoreSnapshot
"""

from risk_engine.heuristic.service import (
    RiskScoreSnapshot,
    compute_risk_score,
    get_historical_calibration_info,
    get_weight_config,
)

__all__ = [
    "compute_risk_score",
    "RiskScoreSnapshot",
    "get_weight_config",
    "get_historical_calibration_info",
]
