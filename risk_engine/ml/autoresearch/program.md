# MEWS autoresearch program

This file adapts the karpathy/autoresearch workflow for MEWS ML risk models.

## Goal

Improve the scalar objective emitted by:

python -m risk_engine.ml.autoresearch.evaluate_candidate --model-type random_forest --split val

Objective definition:

- 70% macro F1 on validation split
- 30% mean crisis high-risk recall across configured crisis windows

Higher is better.

## Allowed edit surface

You may edit these files only:

- risk_engine/ml/config.py
- risk_engine/ml/models.py
- risk_engine/ml/train.py

You must not edit:

- risk_engine/ml/dataset.py label logic
- risk_engine/ml/evaluate.py metric definitions
- data ingestion and non-ML pipeline modules

## Guardrails

- Keep time-series split semantics intact.
- No random shuffling.
- Preserve deterministic defaults (random_state = 42 unless intentionally changed and logged).
- Keep model families interpretable (linear + tree ensembles only).
- Do not introduce external services, secrets, or network calls.

## Experiment loop

1. Read current code and identify one small hypothesis.
2. Make a minimal diff tied to that hypothesis.
3. Run:

python -m risk_engine.ml.autoresearch.evaluate_candidate --model-type random_forest --split val

4. Record objective from stdout and compare with previous result in risk_engine/ml/autoresearch/results.tsv.
5. Keep only objective-improving changes unless there is a strong generalization rationale.
6. Repeat.

## Reporting format

For each run, report:

- hypothesis
- changed file list
- objective delta
- f1_macro delta
- crisis_recall_mean delta

## Baseline run command

python -m risk_engine.ml.autoresearch.evaluate_candidate --model-type random_forest --split val --seed 42 --samples 5000
