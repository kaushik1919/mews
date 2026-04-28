"""Deterministic objective runner for autoresearch-style ML iteration in MEWS.

This script trains one ML model on the existing MEWS mock dataset and emits a
single scalar objective to optimize in autonomous experiment loops.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from risk_engine.ml.config import ModelType
from risk_engine.ml.dataset import create_mock_ml_dataset
from risk_engine.ml.evaluate import EvaluationReport, evaluate_model
from risk_engine.ml.train import train_model

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "timestamp_utc",
    "model_type",
    "split",
    "seed",
    "samples",
    "objective",
    "accuracy",
    "f1_macro",
    "crisis_recall_mean",
]


def _parse_model_type(value: str) -> ModelType:
    try:
        return ModelType(value)
    except ValueError as exc:
        choices = ", ".join(m.value for m in ModelType)
        raise argparse.ArgumentTypeError(
            f"Invalid model type '{value}'. Expected one of: {choices}"
        ) from exc


def _crisis_recall_mean(report: EvaluationReport) -> float:
    if not report.crisis_periods:
        return 0.0
    recalls = [period.high_risk_recall for period in report.crisis_periods]
    return float(np.mean(recalls))


def _objective(report: EvaluationReport) -> float:
    # Weighted objective balances broad classification quality and crisis detection.
    f1_macro = report.classification.f1_macro
    crisis_recall = _crisis_recall_mean(report)
    return 0.7 * f1_macro + 0.3 * crisis_recall


def _append_result(results_tsv: Path, row: dict[str, str]) -> None:
    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_tsv.exists()

    with results_tsv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_once(
    model_type: ModelType,
    split: str,
    seed: int,
    samples: int,
    results_tsv: Path,
) -> tuple[float, EvaluationReport]:
    np.random.seed(seed)

    dataset = create_mock_ml_dataset(n_samples=samples, random_state=seed)

    if dataset.train.n_samples == 0 or dataset.val.n_samples == 0:
        raise ValueError(
            "Sample count is too small for configured time splits. "
            "Use at least 5000 samples or adjust split windows in risk_engine/ml/config.py."
        )

    trained = train_model(model_type=model_type, dataset=dataset, verbose=False)
    report = evaluate_model(trained, dataset=dataset, split=split)

    objective = _objective(report)

    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_type": model_type.value,
        "split": split,
        "seed": str(seed),
        "samples": str(samples),
        "objective": f"{objective:.6f}",
        "accuracy": f"{report.classification.accuracy:.6f}",
        "f1_macro": f"{report.classification.f1_macro:.6f}",
        "crisis_recall_mean": f"{_crisis_recall_mean(report):.6f}",
    }
    _append_result(results_tsv, row)

    return objective, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one autoresearch-style MEWS ML experiment and log its objective."
    )
    parser.add_argument(
        "--model-type",
        type=_parse_model_type,
        default=ModelType.RANDOM_FOREST,
        help="Model family to train.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Evaluation split used for the optimization objective.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mock data generation.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic business-day samples to generate.",
    )
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=Path("risk_engine/ml/autoresearch/results.tsv"),
        help="Path to append experiment metrics in TSV format.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of plain text summary.",
    )
    args = parser.parse_args()

    try:
        objective, report = run_once(
            model_type=args.model_type,
            split=args.split,
            seed=args.seed,
            samples=args.samples,
            results_tsv=args.results_tsv,
        )
    except ImportError as exc:
        logger.error("Missing dependency while training model: %s", exc)
        print("ERROR: Missing ML dependency. Install optional model dependencies first.")
        print(f"DETAIL: {exc}")
        return 2
    except ValueError as exc:
        logger.error("Invalid experiment configuration: %s", exc)
        print("ERROR: Invalid experiment configuration.")
        print(f"DETAIL: {exc}")
        return 2

    confusion_matrix = report.classification.confusion_matrix
    classification_payload = {
        "accuracy": float(report.classification.accuracy),
        "precision_macro": float(report.classification.precision_macro),
        "recall_macro": float(report.classification.recall_macro),
        "f1_macro": float(report.classification.f1_macro),
        "roc_auc_ovr": (
            float(report.classification.roc_auc_ovr)
            if report.classification.roc_auc_ovr is not None
            else None
        ),
        "confusion_matrix": (
            confusion_matrix.tolist() if confusion_matrix is not None else None
        ),
        "per_class_precision": {
            label: float(value)
            for label, value in report.classification.per_class_precision.items()
        },
        "per_class_recall": {
            label: float(value)
            for label, value in report.classification.per_class_recall.items()
        },
    }

    if args.json:
        payload = {
            "objective": float(objective),
            "model_type": args.model_type.value,
            "split": args.split,
            "classification": classification_payload,
            "crisis_periods": [asdict(period) for period in report.crisis_periods],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"OBJECTIVE={objective:.6f}")
        print(f"MODEL={args.model_type.value}")
        print(f"SPLIT={args.split}")
        print(f"F1_MACRO={report.classification.f1_macro:.6f}")
        print(f"ACCURACY={report.classification.accuracy:.6f}")
        print(f"CRISIS_RECALL_MEAN={_crisis_recall_mean(report):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
