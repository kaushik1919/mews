"""
Risk Engine for MEWS-FIN.

Phase 4.0: Heuristic Baseline
Phase 4.x (future): ML-based risk scoring

The risk engine computes a single interpretable risk score
by combining multimodal feature snapshots.

Usage:
    from risk_engine.heuristic import compute_risk_score

    snapshot = compute_risk_score(
        numeric_features={"realized_volatility_20d": 0.25, ...},
        sentiment_features={"news_sentiment_daily": -0.3, ...},
        graph_features={"avg_pairwise_correlation_20d": 0.65, ...},
    )

    print(snapshot.risk_score)  # 0.58
    print(snapshot.regime)      # "HIGH_RISK"
"""

__all__ = ["heuristic"]
