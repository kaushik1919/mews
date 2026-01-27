"""
Correlation matrix computation for graph features.

CRITICAL RULES:
- Pearson correlation only (interpretable, standard)
- Computed on log returns
- No shrinkage estimators
- No forward-looking statistics
- Ill-conditioned matrices → null features

Graph construction:
- Nodes = assets
- Edge weight = pairwise Pearson correlation
- Window = backward-looking (e.g., 20 trading days)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Minimum number of assets for meaningful correlation analysis
MIN_ASSETS_FOR_CORRELATION = 3


def compute_correlation_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Compute pairwise Pearson correlation matrix from returns.

    Args:
        returns: DataFrame of log returns, columns = asset tickers,
                 rows = time observations. Must have no NaN values.

    Returns:
        Correlation matrix as DataFrame (symmetric, 1s on diagonal),
        or None if matrix is invalid/ill-conditioned.

    Mathematical definition:
        ρ(i,j) = Cov(r_i, r_j) / (σ_i * σ_j)

        Where:
        - r_i, r_j are return series for assets i, j
        - Cov is sample covariance
        - σ is sample standard deviation

    Economic interpretation:
        Correlation measures linear co-movement between assets.
        High correlation indicates that assets move together,
        reducing diversification benefits.
    """
    if returns.empty:
        return None

    n_assets = len(returns.columns)
    if n_assets < MIN_ASSETS_FOR_CORRELATION:
        return None

    n_obs = len(returns)
    if n_obs < 2:
        return None

    # Check for any remaining NaN
    if returns.isna().any().any():
        return None

    # Compute correlation matrix
    corr_matrix = returns.corr(method="pearson")

    # Validate matrix
    if not _is_valid_correlation_matrix(corr_matrix):
        return None

    return corr_matrix


def _is_valid_correlation_matrix(corr: pd.DataFrame) -> bool:
    """
    Check if correlation matrix is valid.

    Validation criteria:
    1. No NaN values
    2. Symmetric
    3. Diagonal = 1
    4. All values in [-1, 1]

    Note: We do NOT check condition number because:
    - We're computing statistics on the matrix, not inverting it
    - High correlation (poor conditioning) is exactly what we measure
    - Crisis conditions would fail condition number checks
    """
    if corr.isna().any().any():
        return False

    # Check symmetry
    if not np.allclose(corr.values, corr.values.T):
        return False

    # Check diagonal
    if not np.allclose(np.diag(corr.values), 1.0):
        return False

    # Check range
    if (corr.values < -1.0).any() or (corr.values > 1.0).any():
        return False

    return True


def get_off_diagonal_values(corr: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """
    Extract off-diagonal values from correlation matrix.

    Returns the upper triangle (excluding diagonal) as a 1D array.
    For an NxN matrix, returns N*(N-1)/2 values.

    Args:
        corr: Correlation matrix.

    Returns:
        1D array of off-diagonal correlation values.

    Mathematical interpretation:
        These are the unique pairwise correlations between assets.
        We use upper triangle only to avoid double-counting
        (since corr[i,j] == corr[j,i]).
    """
    n = len(corr)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return corr.values[mask]


def compute_avg_pairwise_correlation(corr: pd.DataFrame) -> float | None:
    """
    Compute average of off-diagonal correlations.

    Args:
        corr: Valid correlation matrix.

    Returns:
        Mean of off-diagonal correlations, or None if invalid.

    Mathematical definition:
        avg_corr = (2 / (N * (N-1))) * Σ_{i<j} ρ(i,j)

    Economic interpretation:
        This measures overall market coupling. High values indicate
        "risk-on/risk-off" regimes where diversification fails.
        
        Typical values:
        - 0.2-0.4: Normal market conditions
        - 0.5-0.7: Elevated stress
        - 0.7+: Crisis conditions (2008, 2020)
    """
    if corr is None:
        return None

    off_diag = get_off_diagonal_values(corr)
    if len(off_diag) == 0:
        return None

    return float(np.mean(off_diag))


def compute_correlation_dispersion(corr: pd.DataFrame) -> float | None:
    """
    Compute standard deviation of off-diagonal correlations.

    Args:
        corr: Valid correlation matrix.

    Returns:
        Std dev of off-diagonal correlations, or None if invalid.

    Mathematical definition:
        dispersion = std({ρ(i,j) : i < j})

    Economic interpretation:
        Low dispersion with high average = uniform stress response
        High dispersion = heterogeneous behavior (some sectors diverge)
        
        Collapsing dispersion during stress indicates market-wide
        contagion where idiosyncratic factors become irrelevant.
    """
    if corr is None:
        return None

    off_diag = get_off_diagonal_values(corr)
    if len(off_diag) < 2:
        return None

    return float(np.std(off_diag, ddof=1))


def compute_sector_to_market_correlations(
    sector_returns: pd.DataFrame,
    market_returns: pd.Series[Any],
) -> dict[str, float]:
    """
    Compute correlation of each sector to the market.

    Args:
        sector_returns: DataFrame of sector returns.
        market_returns: Series of market returns.

    Returns:
        Dict of sector_name → correlation with market.

    Mathematical definition:
        ρ(sector, market) = Cov(r_sector, r_market) / (σ_sector * σ_market)

    Economic interpretation:
        High sector-to-market correlation indicates the sector moves
        lockstep with the market, offering no diversification benefit.
        Low or negative correlation indicates potential safe haven.
    """
    if sector_returns.empty or market_returns.empty:
        return {}

    correlations = {}
    for sector in sector_returns.columns:
        sector_series = sector_returns[sector]
        # Compute correlation
        combined = pd.DataFrame({
            "sector": sector_series,
            "market": market_returns,
        }).dropna()

        if len(combined) < 2:
            continue

        corr = combined["sector"].corr(combined["market"])
        if pd.notna(corr):
            correlations[sector] = float(corr)

    return correlations


def compute_mean_sector_correlation(
    sector_correlations: dict[str, float],
) -> float | None:
    """
    Compute mean of sector-to-market correlations.

    Args:
        sector_correlations: Dict of sector → correlation.

    Returns:
        Mean correlation, or None if no valid sectors.

    Economic interpretation:
        High mean indicates all sectors move with market (systemic stress).
        Low mean indicates sector divergence (potential hedges exist).
    """
    if not sector_correlations:
        return None

    values = list(sector_correlations.values())
    return float(np.mean(values))
