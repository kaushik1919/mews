"""
Temporal Smoothing for Ensemble Risk Scores.

MEWS-FIN Phase 4.2: Apply light temporal smoothing to enhance
stability without masking genuine risk spikes.

Key Constraints (NON-NEGOTIABLE):
    - NO forward-looking smoothing (causal only)
    - Preserve spikes (no heavy damping)
    - Short window only (3-5 days recommended)
    - Document all parameters explicitly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SmoothingMethod(Enum):
    """Supported smoothing methods."""

    EXPONENTIAL_MA = "exponential_ma"  # Exponential moving average
    SIMPLE_MA = "simple_ma"  # Simple moving average
    NONE = "none"  # No smoothing


@dataclass
class SmoothingConfig:
    """
    Configuration for temporal smoothing.

    Attributes:
        method: Smoothing method to use
        window: Window size in trading days (for SMA)
        alpha: Decay factor for EMA (higher = more weight on recent)
        preserve_spikes: If True, don't smooth when current > threshold
        spike_threshold: Absolute score increase to consider a spike
        min_history: Minimum history points required for smoothing
    """

    method: SmoothingMethod = SmoothingMethod.EXPONENTIAL_MA
    window: int = 3  # 3-day window (conservative)
    alpha: float = 0.6  # EMA decay (0.6 = moderate smoothing)
    preserve_spikes: bool = True
    spike_threshold: float = 0.15  # 15% jump considered a spike
    min_history: int = 2

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window < 1:
            raise ValueError(f"Window must be >= 1, got {self.window}")
        if self.window > 10:
            logger.warning(
                f"Large smoothing window ({self.window}) may mask genuine spikes"
            )
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Alpha must be in (0, 1], got {self.alpha}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "window": self.window,
            "alpha": self.alpha,
            "preserve_spikes": self.preserve_spikes,
            "spike_threshold": self.spike_threshold,
            "min_history": self.min_history,
        }


@dataclass
class SmoothingState:
    """
    State for incremental smoothing.

    Used when processing scores one-at-a-time (streaming mode).
    """

    history: list[float] = field(default_factory=list)
    timestamps: list[pd.Timestamp] = field(default_factory=list)
    ema_value: float | None = None
    last_raw: float | None = None

    def add_observation(
        self,
        raw_score: float,
        timestamp: pd.Timestamp,
        config: SmoothingConfig,
    ) -> float:
        """
        Add a new observation and compute smoothed value.

        Args:
            raw_score: Raw (calibrated) risk score
            timestamp: Observation timestamp
            config: Smoothing configuration

        Returns:
            Smoothed risk score
        """
        # Store history
        self.history.append(raw_score)
        self.timestamps.append(timestamp)
        prev_raw = self.last_raw
        self.last_raw = raw_score

        # Not enough history for smoothing
        if len(self.history) < config.min_history:
            self.ema_value = raw_score
            return raw_score

        # Check for spike preservation
        if config.preserve_spikes and prev_raw is not None:
            if raw_score - prev_raw > config.spike_threshold:
                # This is a spike - don't smooth
                self.ema_value = raw_score
                return raw_score

        # Apply smoothing
        if config.method == SmoothingMethod.NONE:
            return raw_score

        elif config.method == SmoothingMethod.EXPONENTIAL_MA:
            return self._compute_ema(raw_score, config)

        elif config.method == SmoothingMethod.SIMPLE_MA:
            return self._compute_sma(config)

        else:
            return raw_score

    def _compute_ema(self, raw_score: float, config: SmoothingConfig) -> float:
        """Compute exponential moving average."""
        if self.ema_value is None:
            self.ema_value = raw_score
        else:
            self.ema_value = config.alpha * raw_score + (1 - config.alpha) * self.ema_value
        return self.ema_value

    def _compute_sma(self, config: SmoothingConfig) -> float:
        """Compute simple moving average."""
        window_scores = self.history[-config.window:]
        return float(np.mean(window_scores))


def apply_temporal_smoothing(
    scores: np.ndarray | list[float],
    timestamps: np.ndarray | list[pd.Timestamp] | None = None,
    config: SmoothingConfig | None = None,
) -> np.ndarray:
    """
    Apply temporal smoothing to a series of risk scores.

    CRITICAL: This is CAUSAL smoothing only - no lookahead.
    Each smoothed value uses only past and current data.

    Args:
        scores: Array of raw risk scores (time-ordered)
        timestamps: Optional timestamps for gap detection
        config: Smoothing configuration

    Returns:
        Array of smoothed risk scores (same length as input)
    """
    if config is None:
        config = SmoothingConfig()

    scores = np.asarray(scores).flatten()
    n = len(scores)

    if n == 0:
        return np.array([])

    if config.method == SmoothingMethod.NONE:
        return scores.copy()

    # Create output array
    smoothed = np.zeros(n)

    if config.method == SmoothingMethod.EXPONENTIAL_MA:
        smoothed = _apply_ema(scores, config)

    elif config.method == SmoothingMethod.SIMPLE_MA:
        smoothed = _apply_sma(scores, config)

    # Preserve spikes if configured
    if config.preserve_spikes:
        smoothed = _preserve_spikes(scores, smoothed, config)

    # Ensure output is in [0, 1]
    smoothed = np.clip(smoothed, 0.0, 1.0)

    return smoothed


def _apply_ema(
    scores: np.ndarray,
    config: SmoothingConfig,
) -> np.ndarray:
    """
    Apply exponential moving average.

    EMA_t = alpha * score_t + (1 - alpha) * EMA_{t-1}

    This is CAUSAL - only uses past values.
    """
    n = len(scores)
    smoothed = np.zeros(n)
    smoothed[0] = scores[0]

    for i in range(1, n):
        smoothed[i] = config.alpha * scores[i] + (1 - config.alpha) * smoothed[i - 1]

    return smoothed


def _apply_sma(
    scores: np.ndarray,
    config: SmoothingConfig,
) -> np.ndarray:
    """
    Apply simple moving average.

    Uses trailing window only (causal).
    """
    n = len(scores)
    smoothed = np.zeros(n)

    for i in range(n):
        start = max(0, i - config.window + 1)
        smoothed[i] = np.mean(scores[start:i + 1])

    return smoothed


def _preserve_spikes(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    config: SmoothingConfig,
) -> np.ndarray:
    """
    Preserve large upward spikes from over-smoothing.

    When raw score jumps significantly, use raw instead of smoothed.
    """
    result = smoothed_scores.copy()

    for i in range(1, len(raw_scores)):
        delta = raw_scores[i] - raw_scores[i - 1]
        if delta > config.spike_threshold:
            # This is a spike - use raw score
            result[i] = raw_scores[i]

    return result


def compute_smoothing_stats(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
) -> dict[str, float]:
    """
    Compute statistics comparing raw vs smoothed scores.

    Useful for monitoring smoothing behavior.

    Args:
        raw_scores: Original scores
        smoothed_scores: Smoothed scores

    Returns:
        Dictionary of statistics
    """
    raw_scores = np.asarray(raw_scores)
    smoothed_scores = np.asarray(smoothed_scores)

    diff = raw_scores - smoothed_scores

    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "raw_std": float(np.std(raw_scores)),
        "smoothed_std": float(np.std(smoothed_scores)),
        "variance_reduction": float(1 - np.var(smoothed_scores) / np.var(raw_scores))
        if np.var(raw_scores) > 0 else 0.0,
        "correlation": float(np.corrcoef(raw_scores, smoothed_scores)[0, 1])
        if len(raw_scores) > 1 else 1.0,
    }
