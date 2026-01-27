"""
MEWS-FIN Numeric Feature Service.

Computes numeric systemic risk features from market prices and volatility indices.
This service is STATELESS - no caching, no persistence, pure functions.

Phase 3.0 scope (from features.yaml):
- Realized volatility (20d, 60d)
- Volatility ratio (20d/60d)
- Max drawdown (20d, 60d)
- Volume z-score (20d)
- Volume-price divergence (20d)
- VIX level (from volatility_indices)

NOT in scope for Phase 3.0:
- Sentiment features (Phase 3.1)
- Graph features (Phase 3.2)
- Macro-derived features (credit_spread_hy, ted_spread)
"""

from .service import NumericFeatureSnapshot, compute_numeric_features

__all__ = ["compute_numeric_features", "NumericFeatureSnapshot"]
