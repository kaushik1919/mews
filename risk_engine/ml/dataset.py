"""
Dataset construction for ML risk models.

CRITICAL RULES:
1. Samples are time-indexed feature snapshots
2. Labels respect publication lags and lookahead constraints
3. Train/val/test splits are time-based (NO random splits)
4. Missing values handled explicitly
5. No feature leakage

Label Construction (for regime classification):
- Use VIX level + realized volatility + drawdown as proxy for regime
- Validate against risk_score.yaml historical anchors
- Labels are assigned based on data available at time t (no lookahead)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from risk_engine.ml.config import (
    ALL_FEATURES,
    ML_CONFIG,
    REGIME_LABELS,
    TargetType,
)


@dataclass
class DatasetSplit:
    """
    A single train/val/test split with features and labels.

    Attributes:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        timestamps: Datetime index for each sample
        feature_names: List of feature column names
        split_name: "train", "val", or "test"
    """

    X: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex
    feature_names: list[str]
    split_name: str

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.y)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1] if len(self.X.shape) > 1 else 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with timestamps as index."""
        df = pd.DataFrame(self.X, columns=self.feature_names, index=self.timestamps)
        df["target"] = self.y
        return df


@dataclass
class MLDataset:
    """
    Complete dataset with train/val/test splits.

    All splits are time-ordered with no overlap.
    """

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    target_type: TargetType
    feature_names: list[str]
    label_mapping: dict[int, str] = field(default_factory=lambda: REGIME_LABELS.copy())

    # Preprocessing state (for inference)
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)


def compute_regime_label(
    vix: float | None,
    realized_vol: float | None,
    drawdown: float | None,
) -> int:
    """
    Compute regime label from available stress indicators.

    LABELING LOGIC (based on risk_score.yaml regime bands):

    EXTREME_RISK (label=3): VIX > 40 OR vol > 0.40 OR drawdown < -0.25
    HIGH_RISK (label=2): VIX > 25 OR vol > 0.25 OR drawdown < -0.15
    MODERATE_RISK (label=1): VIX > 18 OR vol > 0.18 OR drawdown < -0.08
    LOW_RISK (label=0): Otherwise

    Uses OR logic: any single extreme signal elevates regime.
    This is conservative and matches heuristic philosophy.

    Args:
        vix: VIX level (None if missing)
        realized_vol: 20d realized volatility (annualized)
        drawdown: 20d max drawdown (negative value)

    Returns:
        Regime label (0-3)
    """
    # Handle all missing
    if vix is None and realized_vol is None and drawdown is None:
        return 0  # Default to low risk if no data

    # Check EXTREME conditions
    if (vix is not None and vix > 40) or \
       (realized_vol is not None and realized_vol > 0.40) or \
       (drawdown is not None and drawdown < -0.25):
        return 3  # EXTREME_RISK

    # Check HIGH conditions
    if (vix is not None and vix > 25) or \
       (realized_vol is not None and realized_vol > 0.25) or \
       (drawdown is not None and drawdown < -0.15):
        return 2  # HIGH_RISK

    # Check MODERATE conditions
    if (vix is not None and vix > 18) or \
       (realized_vol is not None and realized_vol > 0.18) or \
       (drawdown is not None and drawdown < -0.08):
        return 1  # MODERATE_RISK

    return 0  # LOW_RISK


def compute_binary_crisis_label(regime_label: int) -> int:
    """
    Convert 4-class regime to binary crisis label.

    Crisis = HIGH_RISK or EXTREME_RISK (labels 2-3)
    Normal = LOW_RISK or MODERATE_RISK (labels 0-1)
    """
    return 1 if regime_label >= 2 else 0


def compute_forward_volatility_label(
    returns: pd.Series,
    current_idx: int,
    horizon_days: int,
) -> float | None:
    """
    Compute forward realized volatility as continuous target.

    Args:
        returns: Full returns series
        current_idx: Current position in series
        horizon_days: Forward horizon for volatility calculation

    Returns:
        Annualized forward volatility, or None if insufficient data
    """
    if current_idx + horizon_days >= len(returns):
        return None

    forward_returns = returns.iloc[current_idx + 1 : current_idx + 1 + horizon_days]
    if len(forward_returns) < horizon_days * 0.8:  # Require 80% data
        return None

    # Annualized volatility
    return float(forward_returns.std() * np.sqrt(252))


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Impute missing values in feature matrix.

    Args:
        df: Feature DataFrame
        strategy: "median", "mean", or "zero"
        fill_values: Pre-computed fill values (for val/test sets)

    Returns:
        Tuple of (imputed DataFrame, fill values used)
    """
    if fill_values is None:
        fill_values = {}
        for col in df.columns:
            if strategy == "median":
                fill_values[col] = df[col].median()
            elif strategy == "mean":
                fill_values[col] = df[col].mean()
            else:
                fill_values[col] = 0.0

    result = df.copy()
    for col, value in fill_values.items():
        if col in result.columns:
            result[col] = result[col].fillna(value)

    return result, fill_values


def standardize_features(
    df: pd.DataFrame,
    means: dict[str, float] | None = None,
    stds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Standardize features to zero mean and unit variance.

    Args:
        df: Feature DataFrame
        means: Pre-computed means (for val/test sets)
        stds: Pre-computed stds (for val/test sets)

    Returns:
        Tuple of (standardized DataFrame, means, stds)
    """
    if means is None:
        means = {}
        stds = {}
        for col in df.columns:
            means[col] = df[col].mean()
            stds[col] = df[col].std()
            if stds[col] == 0:
                stds[col] = 1.0  # Avoid division by zero

    result = df.copy()
    for col in df.columns:
        if col in means and col in stds:
            result[col] = (result[col] - means[col]) / stds[col]

    return result, means, stds


def build_dataset(
    feature_data: pd.DataFrame,
    target_type: TargetType | None = None,
    config: Any | None = None,
) -> MLDataset:
    """
    Build ML dataset from feature snapshots.

    Args:
        feature_data: DataFrame with columns = features, index = timestamps
                      Must include 'vix_level', 'realized_volatility_20d',
                      'max_drawdown_20d' for regime labeling
        target_type: Override target type from config
        config: Override ML config

    Returns:
        MLDataset with train/val/test splits

    CRITICAL RULES:
    1. Splits are time-based, no shuffling
    2. Labels computed without lookahead
    3. Standardization fitted on train only
    """
    if config is None:
        config = ML_CONFIG

    if target_type is None:
        target_type = config.target_type

    # Ensure datetime index
    if not isinstance(feature_data.index, pd.DatetimeIndex):
        feature_data.index = pd.to_datetime(feature_data.index)

    # Sort by time (critical for time-series)
    feature_data = feature_data.sort_index()

    # Compute labels
    labels = []
    for idx in range(len(feature_data)):
        row = feature_data.iloc[idx]

        vix = row.get("vix_level")
        vol = row.get("realized_volatility_20d")
        dd = row.get("max_drawdown_20d")

        # Handle NaN
        vix = None if pd.isna(vix) else float(vix)
        vol = None if pd.isna(vol) else float(vol)
        dd = None if pd.isna(dd) else float(dd)

        if target_type == TargetType.REGIME_CLASSIFICATION:
            label = compute_regime_label(vix, vol, dd)
        elif target_type == TargetType.BINARY_CRISIS:
            regime = compute_regime_label(vix, vol, dd)
            label = compute_binary_crisis_label(regime)
        else:  # FORWARD_VOLATILITY
            # Need returns data - not implemented in this version
            # Would require additional data input
            label = 0  # Placeholder

        labels.append(label)

    feature_data = feature_data.copy()
    feature_data["_label"] = labels

    # Time-based splits
    splits = config.splits
    train_mask = (feature_data.index >= splits.train_start) & \
                 (feature_data.index <= splits.train_end)
    val_mask = (feature_data.index >= splits.val_start) & \
               (feature_data.index <= splits.val_end)
    test_mask = (feature_data.index >= splits.test_start) & \
                (feature_data.index <= splits.test_end)

    train_data = feature_data[train_mask].copy()
    val_data = feature_data[val_mask].copy()
    test_data = feature_data[test_mask].copy()

    # Select only valid feature columns
    feature_cols = [c for c in ALL_FEATURES if c in feature_data.columns]

    # Extract features and labels
    X_train = train_data[feature_cols]
    y_train = train_data["_label"].values
    X_val = val_data[feature_cols]
    y_val = val_data["_label"].values
    X_test = test_data[feature_cols]
    y_test = test_data["_label"].values

    # Impute missing values (fit on train)
    strategy = config.missing_value_strategy.replace("impute_", "")
    X_train, fill_values = impute_missing_values(X_train, strategy)
    X_val, _ = impute_missing_values(X_val, strategy, fill_values)
    X_test, _ = impute_missing_values(X_test, strategy, fill_values)

    # Standardize features (fit on train)
    if config.standardize_features:
        X_train, means, stds = standardize_features(X_train)
        X_val, _, _ = standardize_features(X_val, means, stds)
        X_test, _, _ = standardize_features(X_test, means, stds)
    else:
        means, stds = {}, {}

    # Create splits
    train_split = DatasetSplit(
        X=X_train.values,
        y=y_train,
        timestamps=train_data.index,
        feature_names=feature_cols,
        split_name="train",
    )

    val_split = DatasetSplit(
        X=X_val.values,
        y=y_val,
        timestamps=val_data.index,
        feature_names=feature_cols,
        split_name="val",
    )

    test_split = DatasetSplit(
        X=X_test.values,
        y=y_test,
        timestamps=test_data.index,
        feature_names=feature_cols,
        split_name="test",
    )

    return MLDataset(
        train=train_split,
        val=val_split,
        test=test_split,
        target_type=target_type,
        feature_names=feature_cols,
        feature_means=means,
        feature_stds=stds,
    )


def create_mock_dataset(
    n_samples: int = 1000,
    start_date: str = "2005-01-01",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create mock feature data for testing.

    Generates realistic feature distributions with embedded
    crisis periods matching historical anchors.

    Args:
        n_samples: Number of daily samples
        start_date: Start date for time index
        random_state: Random seed

    Returns:
        DataFrame with feature columns and datetime index
    """
    np.random.seed(random_state)

    dates = pd.date_range(start=start_date, periods=n_samples, freq="B")

    # Generate base features with realistic distributions
    data = {}

    # VIX: log-normal, mean ~18, spikes during crises
    vix_base = np.exp(np.random.normal(2.9, 0.3, n_samples))
    data["vix_level"] = np.clip(vix_base, 10, 80)

    # Realized volatility: correlated with VIX
    vol_base = 0.15 + 0.3 * (data["vix_level"] - 18) / 30
    vol_noise = np.random.normal(0, 0.03, n_samples)
    data["realized_volatility_20d"] = np.clip(vol_base + vol_noise, 0.05, 0.80)
    data["realized_volatility_60d"] = (
        data["realized_volatility_20d"] * 0.9 +
        np.random.normal(0, 0.02, n_samples)
    )

    # Volatility ratio
    data["volatility_ratio_20d_60d"] = (
        data["realized_volatility_20d"] / data["realized_volatility_60d"]
    )

    # Drawdown: more negative during high vol
    dd_base = -0.05 - 0.3 * (data["realized_volatility_20d"] - 0.15)
    dd_noise = np.random.normal(0, 0.02, n_samples)
    data["max_drawdown_20d"] = np.clip(dd_base + dd_noise, -0.50, 0)
    data["max_drawdown_60d"] = data["max_drawdown_20d"] * 1.5

    # Volume z-score
    data["volume_zscore_20d"] = np.random.normal(0, 1, n_samples)

    # Volume-price divergence
    data["volume_price_divergence"] = np.random.normal(0, 0.3, n_samples)

    # Sentiment features
    data["news_sentiment_daily"] = np.random.normal(0, 0.3, n_samples)
    data["news_sentiment_5d"] = (
        data["news_sentiment_daily"] * 0.7 +
        np.random.normal(0, 0.1, n_samples)
    )
    data["sentiment_volatility_20d"] = np.abs(
        np.random.normal(0.2, 0.1, n_samples)
    )

    # Graph features
    data["avg_pairwise_correlation_20d"] = np.clip(
        np.random.normal(0.4, 0.15, n_samples), 0, 1
    )
    data["correlation_dispersion_20d"] = np.clip(
        np.random.normal(0.15, 0.05, n_samples), 0, 0.5
    )
    data["sector_correlation_to_market"] = np.clip(
        np.random.normal(0.6, 0.15, n_samples), 0, 1
    )
    data["network_centrality_change"] = np.random.normal(0.05, 0.03, n_samples)

    # Inject crisis periods (elevated VIX and vol)
    crisis_dates = [
        ("2008-09-15", "2009-03-31"),  # GFC
        ("2011-08-01", "2011-10-31"),  # Euro crisis
        ("2015-08-15", "2015-09-15"),  # China
        ("2020-02-20", "2020-04-15"),  # COVID
        ("2022-01-01", "2022-06-30"),  # Rate hikes
    ]

    for crisis_start, crisis_end in crisis_dates:
        crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
        n_crisis = crisis_mask.sum()
        if n_crisis > 0:
            data["vix_level"][crisis_mask] += np.random.uniform(10, 30, n_crisis)
            data["realized_volatility_20d"][crisis_mask] += 0.15
            data["max_drawdown_20d"][crisis_mask] -= 0.10
            data["news_sentiment_daily"][crisis_mask] -= 0.3

    df = pd.DataFrame(data, index=dates)

    # Add some missing values (realistic)
    for col in df.columns:
        missing_mask = np.random.random(len(df)) < 0.02  # 2% missing
        df.loc[missing_mask, col] = np.nan

    return df


def create_mock_ml_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
) -> MLDataset:
    """
    Create mock MLDataset for testing.

    Generates mock feature data and builds complete dataset
    with train/val/test splits. Uses enough samples (5000 business days)
    to cover the full 2005-2024 date range.

    Args:
        n_samples: Number of daily samples (default 5000 to cover ~20 years)
        random_state: Random seed

    Returns:
        MLDataset with train/val/test splits
    """
    df = create_mock_dataset(n_samples=n_samples, random_state=random_state)
    return build_dataset(df)
