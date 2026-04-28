from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from risk_engine.ml.autoresearch import evaluate_candidate
from risk_engine.ml.config import ModelType
from risk_engine.ml.evaluate import (
    ClassificationMetrics,
    EvaluationReport,
    HeuristicComparison,
)


def _make_report() -> EvaluationReport:
    classification = ClassificationMetrics(
        accuracy=0.8,
        precision_macro=0.81,
        recall_macro=0.79,
        f1_macro=0.8,
        roc_auc_ovr=0.85,
        confusion_matrix=np.array([[10, 2], [1, 9]]),
        per_class_precision={"LOW_RISK": 0.83},
        per_class_recall={"LOW_RISK": 0.91},
    )

    heuristic = HeuristicComparison(
        ml_accuracy=0.8,
        heuristic_accuracy=0.78,
        accuracy_delta=0.02,
        ml_crisis_recall=0.75,
        heuristic_crisis_recall=0.7,
        crisis_recall_delta=0.05,
        correlation=0.88,
        mean_absolute_diff=0.07,
        agreement_rate=0.76,
    )

    return EvaluationReport(
        model_type=ModelType.RIDGE,
        split_name="val",
        n_samples=100,
        classification=classification,
        crisis_periods=[],
        lead_time=[],
        heuristic_comparison=heuristic,
    )


def test_append_result_writes_header_once(tmp_path: Path) -> None:
    results = tmp_path / "results.tsv"

    row = {
        "timestamp_utc": "2026-04-28T00:00:00+00:00",
        "model_type": "ridge",
        "split": "val",
        "seed": "42",
        "samples": "5000",
        "objective": "0.900000",
        "accuracy": "0.900000",
        "f1_macro": "0.900000",
        "crisis_recall_mean": "0.900000",
    }

    evaluate_candidate._append_result(results, row)
    evaluate_candidate._append_result(results, row)

    lines = results.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("timestamp_utc\tmodel_type")


def test_run_once_rejects_small_samples(tmp_path: Path) -> None:
    results = tmp_path / "results.tsv"

    try:
        evaluate_candidate.run_once(
            model_type=ModelType.RIDGE,
            split="val",
            seed=42,
            samples=2000,
            results_tsv=results,
        )
    except ValueError as exc:
        assert "too small" in str(exc)
    else:
        raise AssertionError("Expected ValueError for undersized sample count")


def test_main_json_output_has_list_confusion_matrix(monkeypatch, capsys) -> None:
    report = _make_report()

    def _fake_run_once(**_kwargs):
        return 0.8, report

    monkeypatch.setattr(evaluate_candidate, "run_once", _fake_run_once)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_candidate.py",
            "--json",
        ],
    )

    exit_code = evaluate_candidate.main()
    out = capsys.readouterr().out

    payload = json.loads(out)

    assert exit_code == 0
    assert payload["classification"]["confusion_matrix"] == [[10, 2], [1, 9]]


def test_run_once_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    first_results = tmp_path / "first.tsv"
    second_results = tmp_path / "second.tsv"

    first_obj, first_report = evaluate_candidate.run_once(
        model_type=ModelType.RIDGE,
        split="val",
        seed=42,
        samples=5000,
        results_tsv=first_results,
    )
    second_obj, second_report = evaluate_candidate.run_once(
        model_type=ModelType.RIDGE,
        split="val",
        seed=42,
        samples=5000,
        results_tsv=second_results,
    )

    assert first_obj == second_obj
    assert first_report.classification.f1_macro == second_report.classification.f1_macro
