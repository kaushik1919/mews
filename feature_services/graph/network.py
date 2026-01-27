"""
Network centrality computation for graph features.

CRITICAL RULES:
- Use degree centrality only (interpretable, stable)
- No eigenvector centrality (as per spec: "keep it interpretable")
- Edge weights = absolute correlation values
- Centrality change = mean absolute change vs previous window
- No GNNs, no community detection, no graph learning

Network construction:
- Nodes = assets
- Edges = pairwise correlations (absolute values for network)
- Fully connected (all nodes have edges to all others)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Threshold for edge inclusion (correlations below this are weak links)
DEFAULT_EDGE_THRESHOLD = 0.0  # Include all edges by default


def compute_degree_centrality(
    corr: pd.DataFrame,
    use_absolute: bool = True,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> pd.Series[Any]:
    """
    Compute degree centrality for each node in the correlation network.

    Degree centrality = sum of edge weights connected to node,
    normalized by maximum possible (N-1).

    Args:
        corr: Correlation matrix (NxN DataFrame).
        use_absolute: If True, use |ρ| as edge weights. Recommended
                      for measuring connectivity strength.
        edge_threshold: Minimum |ρ| to include an edge. Default 0
                       includes all edges.

    Returns:
        Series of centrality values indexed by asset ticker.
        Values in [0, 1] where 1 = maximally connected.

    Mathematical definition:
        C_D(v) = (1 / (N-1)) * Σ_{u≠v} w(v, u)

        Where w(v, u) = |ρ(v, u)| if use_absolute else ρ(v, u)
        Only edges where |w| >= edge_threshold are included.

    Economic interpretation:
        High centrality = asset strongly correlated with many others.
        These are potential contagion hubs. If they move, the market moves.
        
        Low centrality = more independent asset, potential diversifier.
    """
    if corr is None or corr.empty:
        return pd.Series(dtype=float)

    n = len(corr)
    if n < 2:
        return pd.Series(dtype=float)

    # Get weight matrix
    weights = corr.values.copy()
    if use_absolute:
        weights = np.abs(weights)

    # Apply threshold
    weights[np.abs(weights) < edge_threshold] = 0.0

    # Zero out diagonal (no self-loops)
    np.fill_diagonal(weights, 0.0)

    # Compute degree centrality
    # Sum of weights for each node, normalized by (N-1)
    degree = weights.sum(axis=1) / (n - 1)

    return pd.Series(degree, index=corr.columns)


def compute_centrality_change(
    current_centrality: pd.Series[Any],
    previous_centrality: pd.Series[Any],
) -> float | None:
    """
    Compute mean absolute change in centrality between windows.

    Args:
        current_centrality: Centrality values for current window.
        previous_centrality: Centrality values for previous window.

    Returns:
        Mean absolute change in centrality, or None if insufficient data.

    Mathematical definition:
        Δ_centrality = (1/N) * Σ_v |C_D^t(v) - C_D^{t-1}(v)|

        Where:
        - C_D^t(v) = degree centrality of asset v at time t
        - Only assets present in both windows are included

    Economic interpretation:
        High centrality change = network structure is shifting.
        Assets are becoming more/less connected to the system.
        
        Rapid shifts indicate:
        - New contagion pathways emerging
        - Previous hubs losing influence
        - Regime change in market structure
        
        Stable centrality = consistent market structure.
    """
    if current_centrality.empty or previous_centrality.empty:
        return None

    # Find common assets
    common = current_centrality.index.intersection(previous_centrality.index)
    if len(common) == 0:
        return None

    # Compute absolute change
    current = current_centrality[common]
    previous = previous_centrality[common]
    abs_change = np.abs(current - previous)

    return float(abs_change.mean())


def build_adjacency_from_correlation(
    corr: pd.DataFrame,
    use_absolute: bool = True,
    edge_threshold: float = 0.0,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """
    Build adjacency matrix from correlation matrix.

    This is the weighted adjacency matrix of the correlation network,
    suitable for graph algorithms.

    Args:
        corr: Correlation matrix.
        use_absolute: Use absolute correlations as weights.
        edge_threshold: Minimum weight to include edge.

    Returns:
        NxN numpy array adjacency matrix.

    Graph properties:
        - Undirected (symmetric)
        - No self-loops (diagonal = 0)
        - Edge weight = |ρ| or ρ depending on use_absolute
    """
    if corr is None or corr.empty:
        return np.array([])

    adj = corr.values.copy()
    if use_absolute:
        adj = np.abs(adj)

    # Apply threshold
    adj[np.abs(adj) < edge_threshold] = 0.0

    # Remove self-loops
    np.fill_diagonal(adj, 0.0)

    return adj


def get_network_statistics(
    corr: pd.DataFrame,
    use_absolute: bool = True,
) -> dict[str, float | None]:
    """
    Compute summary statistics of the correlation network.

    Args:
        corr: Correlation matrix.
        use_absolute: Use absolute correlations for edge weights.

    Returns:
        Dict with network statistics:
        - mean_degree: Average degree centrality
        - std_degree: Std dev of degree centrality (inequality measure)
        - max_degree: Maximum degree centrality (most connected node)
        - min_degree: Minimum degree centrality (most independent node)

    Economic interpretation:
        - High mean_degree = tightly coupled market
        - High std_degree = some assets dominate connectivity
        - Low std_degree = uniform coupling (stress everywhere)
    """
    centrality = compute_degree_centrality(corr, use_absolute=use_absolute)

    if centrality.empty:
        return {
            "mean_degree": None,
            "std_degree": None,
            "max_degree": None,
            "min_degree": None,
        }

    return {
        "mean_degree": float(centrality.mean()),
        "std_degree": float(centrality.std(ddof=1)) if len(centrality) > 1 else 0.0,
        "max_degree": float(centrality.max()),
        "min_degree": float(centrality.min()),
    }
