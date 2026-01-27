"""
Graph / Correlation Feature Service for MEWS-FIN.

Phase 3.2: Systemic Connectivity & Stress Features

This module computes graph-based systemic risk features from cross-asset
correlation networks. It measures how tightly the financial system is wired
together, not to predict returns or optimize portfolios.

Features implemented:
- avg_pairwise_correlation_20d: Mean off-diagonal correlation (market coupling)
- correlation_dispersion_20d: Std dev of correlations (heterogeneity measure)
- sector_correlation_to_market: Mean sector-to-market correlation
- network_centrality_change: Mean absolute change in degree centrality

All features are:
- Deterministic (same inputs → same outputs)
- Backward-looking only (no lookahead bias)
- Explainable (can be presented to risk committee)
- Computed from log returns of market prices

Public API:
    from feature_services.graph import compute_graph_features, GraphFeatureSnapshot
"""

from feature_services.graph.service import (
    GraphFeatureSnapshot,
    compute_graph_features,
)

__all__ = [
    "compute_graph_features",
    "GraphFeatureSnapshot",
]
