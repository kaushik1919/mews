"""
CLI Entry Point for Visualization Generation.

MEWS Phase 5B: Generate all documentation figures.

Usage:
    python -m visualization.run_all --mock
    python -m visualization.run_all --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use("Agg")

from .config import ensure_dirs, apply_style
from .architecture import generate_all_architecture_plots
from .features import generate_all_feature_plots
from .risk import generate_all_risk_plots
from .evaluation import generate_all_evaluation_plots


def main() -> int:
    """Main entry point for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate MEWS documentation figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m visualization.run_all --mock
    python -m visualization.run_all --only features
    python -m visualization.run_all --list
        """,
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock/synthetic data (required for demo)",
    )
    
    parser.add_argument(
        "--only",
        choices=["architecture", "features", "risk", "evaluation", "all"],
        default="all",
        help="Generate only specific category of figures",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all figures that would be generated",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("MEWS Documentation Figures:")
        print("\nArchitecture:")
        print("  - figures/architecture/mews_architecture.png")
        print("\nFeatures:")
        print("  - figures/features/volatility_and_drawdown_timeseries.png")
        print("  - figures/features/news_sentiment_vs_market.png")
        print("  - figures/features/avg_correlation_timeseries.png")
        print("\nRisk Engine:")
        print("  - figures/risk_engine/heuristic_risk_score.png")
        print("  - figures/risk_engine/ml_vs_heuristic_comparison.png")
        print("  - figures/risk_engine/calibration_curve.png")
        print("  - figures/risk_engine/ensemble_vs_components.png")
        print("  - figures/risk_engine/shap_global_importance.png")
        print("\nEvaluation:")
        print("  - figures/evaluation/lead_time_bar_chart.png")
        print("  - figures/evaluation/false_positive_rate_by_threshold.png")
        print("  - figures/evaluation/false_alarm_duration.png")
        print("\nDemo:")
        print("  - figures/demo/daily_risk_snapshot.png")
        return 0
    
    # Require --mock for now (real data not yet implemented)
    if not args.mock:
        print("Error: --mock flag is required (real data generation not yet implemented)")
        return 1
    
    # Ensure output directories exist
    ensure_dirs()
    
    # Apply consistent style
    apply_style()
    
    generated: list[Path] = []
    
    try:
        if args.only in ("all", "architecture"):
            if args.verbose:
                print("Generating architecture figures...")
            paths = generate_all_architecture_plots(mock=args.mock)
            generated.extend(paths)
            if args.verbose:
                for p in paths:
                    print(f"  [OK] {p}")
        
        if args.only in ("all", "features"):
            if args.verbose:
                print("Generating feature figures...")
            paths = generate_all_feature_plots(mock=args.mock)
            generated.extend(paths)
            if args.verbose:
                for p in paths:
                    print(f"  [OK] {p}")
        
        if args.only in ("all", "risk"):
            if args.verbose:
                print("Generating risk engine figures...")
            paths = generate_all_risk_plots(mock=args.mock)
            generated.extend(paths)
            if args.verbose:
                for p in paths:
                    print(f"  [OK] {p}")
        
        if args.only in ("all", "evaluation"):
            if args.verbose:
                print("Generating evaluation figures...")
            paths = generate_all_evaluation_plots(mock=args.mock)
            generated.extend(paths)
            if args.verbose:
                for p in paths:
                    print(f"  [OK] {p}")
        
        print(f"\nGenerated {len(generated)} figures successfully")
        
        # Verify all expected files exist
        if args.only == "all":
            expected_count = 13  # 2 arch + 3 feat + 5 risk + 3 eval
            if len(generated) != expected_count:
                print(f"Warning: Expected {expected_count} figures, got {len(generated)}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
