# AIML-01: Real-Time Financial Fraud Detection Platform

Final-year B.Tech CSE (AI/ML) capstone. CPU-only, Windows-friendly. **Read `docs/final_audit_report.md` first: it states
exactly what works, what was only written, and what has never been run.**

> **Pass 6 status:** the real-data objective (PaySim -> XGBoost -> artifact -> API -> dashboard) is **BLOCKED in the authoring environment** (no network, no dataset, no xgboost); no real model or metrics exist and no artifact is shipped. The workflow, tools and exact commands are in place and verified on synthetic data: see `docs/windows_setup.md`, `docs/real_training_report.md`.
>
> **Earlier honest status (30 Oct 2026):** the code base is complete for a local demo, but it has never been run on the real
> PaySim dataset, the FastAPI routes and React dashboard have never been started/built (the authoring environment had
> no network), and XGBoost / Kafka / Feast / MLflow / Evidently were never executed. All verified results come from
> synthetic test fixtures. No model metrics from real data exist yet.

## What it does
Score transactions with a gradient-boosting model, store results in SQLite, and review them in an analyst dashboard
(alerts, cases, analyst labels, drift monitoring, controlled retraining workflow). Scores are **uncalibrated ranking
scores**, never presented as probabilities. There is **no authentication**: run on localhost only.

| Area | State |
|---|---|
| Dataset inspection / EDA (`ml/eda`) | Real code from the start; **not run on real data** |
| Schema, leakage-safe features, time split, training, evaluation, artifacts (`ml/`) | Implemented, synthetic-tested; XGBoost path written, **not executed** |
| FastAPI service (`backend/`) | Core logic tested; HTTP layer **never started** |
| React dashboard (`frontend/`) | Logic tested + type-checked; **never built or opened** |
| Replay / Kafka (`streaming/`) | API replay tested against a local fake API; Kafka logic tested with fakes; **no broker** |
| Monitoring + retraining workflow (`monitoring/`) | Implemented, synthetic-tested; Evidently report **unverified** |
| Feast (`feature_repo/`) | Config + consistency tests without Feast; **never applied** |
| Graph (`ml/graph/`) | Structure analysis + baseline only; **no GNN** |
| Docker / Compose / Kubernetes / CI | Files written, YAML-validated; **never built or run** |

## Quick start (Windows PowerShell) - details in `docs/windows_setup.md`
```
cd C:\Projects\AIML-01-Fraud-Detection
.\.venv-test\Scripts\python.exe -m pip list                          # what is already installed
.\.venv-test\Scripts\python.exe -m pip install -r requirements-dev.txt   # installs only what is missing
.\.venv-test\Scripts\python.exe -m pytest -q                         # run the tests
```
No dataset yet? Rehearse with clearly-labelled synthetic data (never report its numbers):
```
.\.venv-test\Scripts\python.exe scripts\make_synthetic_demo.py
.\.venv-test\Scripts\python.exe -m ml.train --data ml\data\raw\SYNTHETIC_demo.csv --model hist_gbm --no-mlflow --run-id demo --data-label "SYNTHETIC demo data"
```
With the real PaySim CSV in `ml\data\raw\`: `docs/windows_setup.md` §3-8 (EDA `python -m ml.eda.analyze`, split check `python -m ml.splits`, then `python -m ml.train --model xgboost`; compare `--feature-set base|pre_balance|all_balance` with `python -m ml.compare_runs`).

## Documentation
| File | Contents |
|---|---|
| `docs/final_audit_report.md` | Implemented / partial / missing, test results, limitations, next actions, demo readiness |
| `docs/windows_setup.md` | Step-by-step PowerShell guide (check env, tests, EDA, train, API, frontend, stop) |
| `docs/architecture.md` | Components, data flow, training/serving contract |
| `docs/paysim_provenance.md` | PaySim source, terms (to fill), acquisition steps, limitations (`dataset.md` points here) |
| `docs/eda_report.md`, `docs/feature_engineering.md` | Generated EDA (from the real file) and every feature, by feature set |
| `docs/real_training_report.md`, `docs/model_evaluation.md`, `docs/reproducibility.md`, `docs/model_card.md` | Training record template, evaluation protocol, reproduction steps, model card |
| `docs/git_workflow.md` | Safe Git commands and model-artifact storage strategy |
| `docs/model_limitations.md` | Evaluation protocol, threshold policy, imbalance, risks |
| `docs/api.md` | Endpoints, behaviour, run commands |
| `docs/frontend.md`, `docs/streaming.md`, `docs/monitoring.md`, `docs/feast.md`, `docs/graph.md` | Per-component status |
| `docs/deployment.md` | Docker, Compose, Minikube (unverified) and limits |
| `docs/security_review.md` | Controls, risks, what was and was not checked |
| `docs/testing_report.md` | Every check as PASS / FAIL / BLOCKED / NOT RUN with commands |
| `docs/demo_script.md` | 10-minute presentation using only implemented functionality |
| `docs/requirements_checklist.md` | Assignment requirement-by-requirement status |

## Layout
```
backend/app/    FastAPI app factory, service layer, SQLite repository, validation, rate limiter
ml/             eda/ (inspection), pipeline/ (schema, features, split, models, evaluate, artifacts), train.py, graph/, feast_support.py
monitoring/     PSI drift, data quality, labelled performance, optional Evidently adapter, CLI
streaming/      replay producer/consumer, outcome recording (python -m streaming)
frontend/       React + TypeScript + Vite dashboard
feature_repo/   Feast definitions (optional)
deploy/k8s/     Kubernetes example manifests; Dockerfile, docker-compose.yml, .github/workflows/ci.yml at the root
scripts/        make_synthetic_demo.py
tests/          synthetic-fixture tests (no real data needed)
```
Note: `ml/inspect_dataset.py` is an older duplicate of `ml/eda/inspect_dataset.py` (kept because it has tests/history);
prefer the `ml/eda` version.

## Rules this project follows
Datasets, secrets, `.env`, databases, virtual environments, model artifacts, `node_modules` and build output are git-ignored.
`.env.example` holds placeholders only (the app reads real environment variables; it does not load `.env` automatically).
No commits or pushes were made on your behalf.
