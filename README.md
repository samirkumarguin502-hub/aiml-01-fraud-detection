# AIML-01 Real-Time Financial Fraud Detection Platform

Internship capstone (batch start 20 Oct 2026, deadline **20 Nov 2026**).
Simulated/replayed transactions only. **No integration with any real financial institution.**

## Status (updated honestly each session)
| Area | Status |
|---|---|
| Repo skeleton, .gitignore, .env.example | Done |
| Dataset inspection + EDA report scripts | Written; unit-tested on synthetic data only |
| Real dataset downloaded and inspected | **Not done - must be downloaded manually (Kaggle login)** |
| EDA report from real data | **Not generated** |
| Everything else (features, XGBoost, API, Kafka, GNN, dashboard, Feast, K8s) | Not started |

No model results exist yet. Nothing in this repo should be read as a measured result until
it appears in `docs/eda/` or MLflow from a real run.

## Quick start (Windows PowerShell)
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
1. Download PaySim (see `docs/dataset.md`) and put the CSV in `ml/data/raw/`.
2. `python ml/eda/inspect_dataset.py`
3. `python ml/eda/make_eda_report.py`
4. `python -m pytest tests -q`

## Layout
`ml/` data + EDA + training, `backend/` FastAPI, `streaming/` Kafka replay, `feature_repo/` Feast,
`frontend/` React, `deploy/` Docker/K8s, `tests/`, `docs/`. Empty folders are placeholders.
