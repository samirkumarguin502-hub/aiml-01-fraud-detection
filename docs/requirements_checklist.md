# Requirements checklist (status is evidence-based)
Legend: [ ] not started, [~] partial, [x] done and verified

## Step 1 Dataset and EDA
- [~] Dataset documentation (`docs/dataset.md`; licence and real stats pending)
- [~] EDA scripts written and synthetic-tested; **real-data run pending**
- [~] Missing/duplicate/class-imbalance checks (implemented, not yet run on real data)
- [ ] Time-based train/val/test split
- [ ] Leakage prevention decision (balance columns) from real EDA
- [ ] Reproducible preprocessing
## Step 2 Features + Feast        [ ]
## Step 3 XGBoost + MLflow        [ ]
## Step 4 Graph + PyG GNN         [ ]
## Step 5 Kafka + FastAPI         [ ]
## Step 6 Evidently + monitoring  [ ]
## Step 7 Dashboard + Docker/K8s  [ ]
## Deliverables: demo, README, diagram, API docs, results, tests, screenshots, presentation, reflection  [ ]

## Test evidence
- 5 unit tests for inspection/report code: passed via a plain-Python runner on 28 Oct 2026 in the
  assistant's sandbox (pytest was unavailable there). Run `python -m pytest tests -q` locally to confirm.
