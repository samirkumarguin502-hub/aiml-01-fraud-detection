# Remaining Work

This file records unfinished work only. Do not repeat already completed tests.

## Verified
- Project test environment: `.venv-test` uses Python 3.12.10.
- Node.js 24.22.1 and npm 11.19.1 are available.
- Baseline Python suite before the fixture isolation fix: 174 passed, 1 failed, 1 skipped.
- The baseline failure was a shared replay-summary path leaking into a temporary service; fixed by giving the temporary service an isolated outcomes directory.
- Focused pipeline-status test after the fix: 1 passed.
- FastAPI HTTP suite: 13 passed.
- Existing `pre-balance-full-v2` XGBoost artifact reload and one-row prediction succeeded with threshold 0.8197945356369019.
- Graph/Feast suite: 10 passed after adding the optional CPU GraphSAGE implementation.
- Final full Python suite: 175 passed, 1 skipped, 1 warning.
- XGBoost and MLflow are installed in `.venv-test`.
- A real PaySim CSV and existing XGBoost artifacts are present under `ml/data/raw/paysim dataset.csv` and `ml/artifacts/`.
- The existing pre-balance-v2 primary artifact is XGBoost and contains real test/validation metrics and reload metadata.
- Frontend typecheck, 27 library tests, and production build passed in the last run.
- One component test timed out under the environment's 5-second default before the test runner itself later failed to start workers; no assertion or timeout value should be changed to force this result.

## Implement now
1. Complete focused verification of MLflow fake-run logging and optional API-key enforcement.
2. Keep optional PyG dependencies lazy; document exact install commands and CPU-only execution.
3. Verify Feast, Evidently and Kafka paths only if their optional libraries/broker are actually available.
4. Validate Docker Compose and Kubernetes manifests with locally available tooling.
5. At final verification, run the full Python suite once and report actual pass/fail counts.

## Known infrastructure gaps
- PyTorch and PyTorch Geometric are not installed in `.venv-test`.
- Feast, Evidently and `confluent-kafka` are not installed in `.venv-test`.
- Docker, Kubernetes and cloud infrastructure availability must be reported only after direct verification.

## Latest implementation
- MLflow logging now creates run metadata, params, numeric metrics and generated artifacts in the correct run lifecycle; a fake-module test verifies the contract without external infrastructure.
- FastAPI supports optional `API_KEY` authentication via Bearer or `X-API-Key`; when unset, behavior remains unchanged, and health/readiness stay public.
- Docker Compose passes optional `API_KEY` through from `.env`.
- The XGBoost GitHub Actions job is no longer marked non-blocking.
- Focused API auth/health verification: 2 passed.
- Focused MLflow/training artifact verification: 2 passed.
- Final full Python suite: 177 passed, 1 skipped, 1 warning.

## Deployment verification status
- Docker CLI is installed (`29.9.2`), but Compose validation did not return while the Docker daemon was unavailable.
- `kubectl` and Minikube clients are installed (`v1.36.1`/`v1.39.0`), but cluster validation was blocked by a local TLS API endpoint that timed out; manifests have not been claimed deployed.
