"""
Daily Pipeline Configuration.

MEWS-FIN Phase 5A: Runtime configuration for daily risk pipeline.

All configuration is explicit and overridable. No magic defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for data ingestion step."""

    # Data sources to ingest (in order)
    sources: tuple[str, ...] = (
        "market_prices",
        "volatility_indices",
        "macro_rates",
        "financial_news",
    )

    # Paths for data outputs
    output_base_path: str = "data_ingestion/outputs/datasets"

    # Version tag for dataset outputs
    version: str = "v1"


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature computation step."""

    # Feature services to run (in order)
    services: tuple[str, ...] = (
        "numeric",
        "sentiment",
        "graph",
    )

    # Window sizes
    volatility_window_days: int = 20
    sentiment_window_days: int = 5
    correlation_window_days: int = 20


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk scoring step."""

    # Models to run
    run_heuristic: bool = True
    run_ml: bool = True
    run_ensemble: bool = True

    # ML model selection
    primary_ml_model: str = "random_forest"
    secondary_ml_model: str | None = "xgboost"

    # Ensemble settings
    use_calibration: bool = True
    use_smoothing: bool = True

    # Thresholds for regime mapping (from risk_score.yaml)
    regime_thresholds: dict[str, float] = field(default_factory=lambda: {
        "LOW_RISK": 0.0,
        "MODERATE_RISK": 0.25,
        "HIGH_RISK": 0.50,
        "EXTREME_RISK": 0.75,
    })


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for report generation."""

    # Report format
    show_sub_scores: bool = True
    show_dominant_factors: bool = True
    show_model_breakdown: bool = True
    show_regime_rationale: bool = True

    # Number of top factors to show
    top_factors_count: int = 3

    # Output settings
    output_to_file: bool = False
    output_path: str = "reports"


@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration.

    All settings are explicit. No hidden state.
    """

    # Run mode
    mock_mode: bool = False

    # Component configs
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Pipeline metadata
    version: str = "5.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mock_mode": self.mock_mode,
            "version": self.version,
            "ingestion": {
                "sources": self.ingestion.sources,
                "output_base_path": self.ingestion.output_base_path,
                "version": self.ingestion.version,
            },
            "features": {
                "services": self.features.services,
                "volatility_window_days": self.features.volatility_window_days,
                "sentiment_window_days": self.features.sentiment_window_days,
                "correlation_window_days": self.features.correlation_window_days,
            },
            "risk": {
                "run_heuristic": self.risk.run_heuristic,
                "run_ml": self.risk.run_ml,
                "run_ensemble": self.risk.run_ensemble,
                "primary_ml_model": self.risk.primary_ml_model,
                "secondary_ml_model": self.risk.secondary_ml_model,
                "use_calibration": self.risk.use_calibration,
                "use_smoothing": self.risk.use_smoothing,
            },
            "report": {
                "show_sub_scores": self.report.show_sub_scores,
                "show_dominant_factors": self.report.show_dominant_factors,
                "show_model_breakdown": self.report.show_model_breakdown,
                "show_regime_rationale": self.report.show_regime_rationale,
                "top_factors_count": self.report.top_factors_count,
            },
        }


# Default configuration
DEFAULT_CONFIG = PipelineConfig()

# Mock mode configuration (for CI and demos)
MOCK_CONFIG = PipelineConfig(mock_mode=True)
