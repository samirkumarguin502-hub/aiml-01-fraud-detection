# Backend API (pass 3)

Status: the framework-independent core (validation, CSV parsing, SQLite repository, rate limiter, model
loading, service layer) is **tested**. The FastAPI binding (`backend/app/main.py`) is **written but has not been
executed anywhere yet**; its HTTP tests (`tests/test_backend_http.py`) skip until fastapi + httpx are installed.
Prediction has only been exercised with an artifact trained on **synthetic** data.

## Run (Windows PowerShell, from the project root, existing venv)
```
.\.venv-test\Scripts\python.exe -m pip list                      # see what is already installed
.\.venv-test\Scripts\python.exe -m pip install fastapi uvicorn httpx   # only what is missing (pydantic must be v2)
$env:MODEL_ARTIFACT_DIR = ".\ml\artifacts\<run_id>\primary"       # optional: omit to run without a model
.\.venv-test\Scripts\python.exe -m uvicorn backend.app.main:create_app --factory --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs (disabled when `APP_ENV=production`). Without a model the API still starts;
`/api/v1/predict*` returns 503 and `/api/v1/ready` returns 503 until a valid artifact is configured.
Tests: `.\.venv-test\Scripts\python.exe -m pytest tests -q`

## Endpoints (all under `/api/v1`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: process is up (always 200) |
| GET | `/ready` | Readiness: 200 only if SQLite is reachable AND a valid model is loaded, else 503 with reasons |
| GET | `/model` | Model state, provenance, threshold, and stored metrics with their origin label |
| POST | `/predict` | Score one transaction (JSON). 503 if no model. Persists the result |
| POST | `/predict/batch` | Score a CSV sent as the raw body (`Content-Type: text/csv`). Persists all rows atomically |
| GET | `/transactions` | Paginated history; filters: `flagged,type,min_score,max_score,source,case_status,batch_id`; `sort` in `id,created_utc,score,amount,step` |
| GET | `/transactions/{id}` | Detail incl. case + feedback + case history |
| PATCH | `/transactions/{id}/case` | Set case status (`open,in_review,confirmed_fraud,false_positive,closed`) + optional note |
| PUT | `/transactions/{id}/feedback` | Analyst label `1` fraud / `0` not fraud (+ comment); replaces the previous label |
| GET | `/alerts` | Flagged transactions (score >= validation-selected threshold), filterable by case status |
| GET | `/dashboard/summary` | Counts, score histogram, per-day and per-type counts, case/feedback counts, model state |

Single prediction body: `{"step":5,"type":"TRANSFER","amount":1234.5,"nameOrig":"C1","nameDest":"C2"}`.
Unknown fields (including balance columns) are rejected on JSON. In CSV uploads extra columns are ignored:
never parsed, stored or logged. Errors always look like `{"error":{"code","message","details?","request_id"}}`
and never echo submitted values.

```
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict -H "Content-Type: application/json" `
  -d '{\"step\":5,\"type\":\"TRANSFER\",\"amount\":1234.5,\"nameOrig\":\"C1\",\"nameDest\":\"C2\"}'
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict/batch -H "Content-Type: text/csv" --data-binary "@transactions.csv"
```

## Behaviour you should know
- **Scores are uncalibrated ranking scores**, not probabilities; every prediction response says so. "Flagged"
  only means score >= the threshold chosen on validation data.
- **Model loading** happens once at startup from `MODEL_ARTIFACT_DIR`. States: `unavailable` (nothing
  configured/found), `invalid` (corrupt, tampered, or version-incompatible: SHA-256 and library checks),
  `ready`. There is no silent fallback model. The path must point at one specific artifact; nothing is
  auto-selected, so nothing is "promoted" by accident.
- **Metrics** shown by `/model` come from the artifact's own metadata, with an origin label. If the artifact was
  trained on synthetic data the response sets `metrics_are_synthetic: true`. Dashboard counts come only from
  transactions scored through this API; the dashboard is empty until something is scored.
- **History features / train-serve skew:** `orig_prev_*` features use only the sender's earlier transactions
  previously scored by this API (plus earlier rows in the same CSV), not the dataset's full history.
- **Persistence:** SQLite file (`DATABASE_URL`, default `fraud_platform.db`, git-ignored). Stores submitted
  fields, score, model hash and data label. Each write is one transaction with rollback. Uploaded CSVs are read
  into memory with a hard cap and are **never written to disk** (no temp files to clean up).

## Security status (honest)
Implemented and tested at the service level: strict input validation, size/row limits, parametrised SQL with
whitelisted sort columns, no value echo in errors, log whitelist (no transaction fields in logs), in-memory
rate limiting (single process only; resets on restart; use a gateway for real protection), restrictive CORS
(explicit origins, no credentials, `*` rejected at startup), no stack traces in responses, docs off in production.
Implemented but only covered by not-yet-run HTTP tests: CORS middleware, security headers, 429/413 responses.

**Not implemented: authentication and authorization.** Anyone who can reach the port can score transactions and
change case status. Keep `--host 127.0.0.1`. No threat model or external security review has been done; this is
not production-secure. JSON bodies without a Content-Length header are not size-checked by the middleware.

## Feature sets and request fields (Pass 6)
`GET /api/v1/model` returns `model.feature_set` and `model.input_fields`. A `base` model accepts exactly `step, type, amount, nameOrig, nameDest`; a `pre_balance` model additionally requires `oldbalanceOrg` and `oldbalanceDest`; only a diagnostic `all_balance` model accepts `newbalanceOrig`/`newbalanceDest`. JSON requests with fields the model does not use are rejected; CSV uploads must contain the required columns and other columns are ignored. Existing databases are migrated automatically (nullable balance columns).
