# Testing report (Pass 6, 30 Oct 2026, authoring environment)

Environment: Linux sandbox, Python 3.12.3, Node 22.22.2, **no network** (kaggle.com and PyPI unreachable), no pytest, no fastapi/pydantic/uvicorn/httpx,
no xgboost/mlflow, no Docker/kubectl, **no real dataset**. matplotlib and psutil were available, so plots and memory measurement were genuinely exercised.
"Custom runner" = a ~20-line script that imports the test modules and calls each `test_*` function (supports `tmp_path`; reports `unittest.SkipTest` as skipped).
**It is not pytest; real pytest has never been run on this code.** Status: PASS / FAIL / BLOCKED (cannot run here) / NOT RUN.

## Pass 6 primary objective: BLOCKED
| Step | Status | Reason |
|---|---|---|
| Acquire the real PaySim CSV | **BLOCKED** | no network; Kaggle needs login and no credentials exist. Manual steps: `docs/windows_setup.md` §3 |
| Inspect/EDA on the real file | **BLOCKED** | no dataset |
| Train XGBoost on real data, real metrics, real artifact | **BLOCKED** | no dataset; xgboost not installed. **No real model, metrics or artifact exist; none is included in the ZIP** |
| API inference with the real artifact; React with real responses | **BLOCKED** | no artifact; FastAPI/Node packages not installable |
| MLflow run | **BLOCKED** | mlflow not installed |

## A. Python tests, synthetic fixtures only
| # | Check | Command | Status | Result |
|---|---|---|---|---|
| A1 | All Python tests | `python3 /tmp/mini_runner.py tests.test_inspect_dataset tests.test_pipeline_core tests.test_training_pipeline tests.test_backend_core tests.test_backend_http tests.test_monitoring tests.test_streaming tests.test_feast_graph tests.test_feature_sets tests.test_eval_and_tools` (+ `tests.test_docs_consistency`) | PASS | 162 passed, 0 failed, 14 skipped |
| A2 | Per module | same runner | PASS | inspect 5 · pipeline core 11 · training 27 (+1 skip) · backend core 39 · backend HTTP 0 (13 skipped) · monitoring 14 · streaming 16 · feast/graph 10 · feature sets 17 · eval & tools 21 · docs consistency 2 |
| A3 | Real pytest | `python -m pytest -q` | **BLOCKED** | pytest not installed; `pytest.ini` exists but was never exercised |
| A4 | FastAPI HTTP tests (13) | `tests/test_backend_http.py` | **BLOCKED** | fastapi/httpx missing: 13 skipped |
| A5 | Real XGBoost end-to-end | `test_real_xgboost_end_to_end` | **BLOCKED** | xgboost missing: 1 skipped (a stub test checks call arguments only) |
| A6 | Compile check | `python3 -m compileall -q backend ml monitoring streaming tests scripts` | PASS | no errors |

### Required test topics (Pass 6 list) and where they are covered (all synthetic)
| Topic | Tests | Status |
|---|---|---|
| Dataset schema; missing/invalid columns | `test_pipeline_core`, `test_feature_sets` (schema per feature set, `load_dataset`), `test_training_pipeline` | PASS (synthetic) |
| Feature engineering; leakage prevention | `test_pipeline_core`, `test_feature_sets` (exact feature lists, pre-balance values, post-transaction columns and labels ignored, causality) | PASS (synthetic) |
| Chronological splitting | `test_pipeline_core`, `test_training_pipeline`, `test_eval_and_tools` (`ml.splits`, unusable-split detection) | PASS (synthetic) |
| Model training; hyperparameters; search on validation only | `test_training_pipeline`, `test_eval_and_tools` | PASS with scikit-learn models; XGBoost **BLOCKED** |
| Serialization, reload, prediction consistency | `test_training_pipeline`, `test_eval_and_tools` (reload comparison, failure aborts and cleans up, no training-only imports needed) | PASS with scikit-learn models; XGBoost **BLOCKED** |
| Threshold handling | `test_training_pipeline` (`select_threshold`, validation-only) | PASS |
| Evaluation outputs: classification report, curves, importance, plots, timing/memory | `test_eval_and_tools` | PASS (PNG files verified by header) |
| API single/batch prediction (service level) incl. balance models, DB migration | `test_backend_core`, `test_feature_sets` | PASS |
| API over HTTP | `test_backend_http` | **BLOCKED** |
| Model-not-ready behaviour | `test_backend_core`, `test_monitoring` | PASS (service level) |
| Documentation matches code (every feature listed) | `test_docs_consistency` | PASS |

## B. CLI smoke tests on synthetic CSV, scratch copy of the project (results meaningless)
| # | Command | Status | Result |
|---|---|---|---|
| B1 | `python -m ml.splits --data SYNTHETIC_demo.csv` | PASS | 3 usable splits, steps 1-72 / 73-96 / 97-120 |
| B2 | `python -m ml.eda.analyze` | PASS | report + `balance_analysis.json` + 4 plots written |
| B3 | `python -m ml.train --feature-set base\|pre_balance\|all_balance --model hist_gbm` | PASS | 3 runs; reload check passed (max abs diff 0.0); plots written |
| B4 | `python -m ml.train --search` | PASS | 4 validation trials, best chosen, test not used |
| B5 | `python -m ml.compare_runs <3 runs>` | PASS | table printed |
| B6 | load artifact via `load_artifact` / `python -m monitoring status` | PASS | metadata and `insufficient_data` state as expected |
| B7 | earlier-pass CLIs (xgboost-missing error, streaming API-unreachable, graph analyze) | PASS | clean errors/outputs |
Perfect synthetic metrics reflect the fixture's planted pattern only.

## C. Frontend
| # | Check | Command | Status | Result |
|---|---|---|---|---|
| C1 | Logic tests (client, guards, validation incl. balance fields, CSV, formatting, errors) | `node --test "src/lib/*.test.ts"` | PASS | 27 passed, 0 failed |
| C2 | Strict type-check, logic modules | `tsc` | PASS | clean |
| C3 | Type-check, components | `tsc` against a hand-written React shim | PASS (weak) | clean; real `@types/react` unavailable |
| C4 | Component tests (Vitest + Testing Library; updated for balance fields and evaluation charts) | `npm run test:components` | **NOT RUN** | cannot install packages |
| C5 | `npm install`, `npm run build`, dev server | | **BLOCKED** | no network. The UI has never been rendered |

## D. Deployment and security (unchanged this pass)
YAML parse of Compose/CI/K8s: PASS. `docker build`, `kubectl`, CI run: BLOCKED / NOT RUN. Dependency vulnerability scan: BLOCKED. Secret/dangerous-pattern scan, forbidden-file scan of the ZIP and
`.gitignore` effectiveness on a throwaway repo: PASS (see `docs/final_audit_report.md` §8).

## Defects found during Pass 6
A model-metadata edit silently did not apply (the line was formatted differently than assumed); the first run of the new tests exposed it (`/model` lacked `input_fields`) and it was fixed and
verified. A test expectation of mine was wrong (steps with zero fraud) and was corrected, not the code.

## Commands to run on your machine
`& $py -m pytest -q` · `cd frontend; npm install; npm run typecheck; npm run test:lib; npm test; npm run build` · then `docs/windows_setup.md` §3-§12.
