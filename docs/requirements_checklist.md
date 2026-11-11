# Requirements checklist (status as of 30 Oct 2026)

Legend: **T** = implemented and tested here on synthetic data · **W** = written, never executed · **R** = needs the real dataset/your machine · **-** = not implemented.

## Machine learning
| Requirement | Status |
|---|---|
| PaySim with documented provenance, usage terms | **R**: provenance text exists; licence, access date, SHA-256 and real stats must come from your file (`docs/dataset.md`) |
| Dataset inspection and EDA | **T** on synthetic CSV; **R** on real data |
| Leakage-aware features; time-aware split | **T** (balance columns and `isFlaggedFraud` never loaded; strictly-earlier-step history; whole-step split) |
| XGBoost with imbalance handling | **T** in an existing real-file artifact; `.venv-test` executes the real XGBoost reload path |
| Precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, threshold analysis | **T**. Per-class "classification report" text not produced |
| Decision threshold + imbalance consequences documented | **T** (`docs/model_limitations.md`, report.md) |
| MLflow tracking | **W**; the "mlflow missing" fallback is **T** |
| Same preprocessing in training and inference | **T** (one `FeatureContract`; consistency tests) |
| No fabricated results; no test-set tuning | enforced in code/tests; **no real metrics exist yet** |
## Backend
| FastAPI API, health, validation, schemas, docs | Core logic **T**; HTTP layer **W** |
| Single + batch CSV prediction | **T** (service level); HTTP **W** |
| Safe missing-model / malformed input / exception handling; no secrets/paths in responses | **T** (service level); HTTP handlers **W** |
## Streaming and features
| Kafka replay/streaming | API-mode replay **T** (fake API); Kafka **W** (fake-client tests, no broker) |
| Feast | consistency logic **T**; store **W** (never applied) |
| Evidently drift monitoring | own PSI monitoring **T**; Evidently adapter **W** |
| PyTorch Geometric fraud rings (optional) | **W/T** (GraphSAGE training path implemented with lazy optional dependencies; package execution not available in `.venv-test`) |
| Optional pieces identifiable; baseline runs without them | **T** |
| No claim of real-time production streaming | documented everywhere |
## Frontend
| Responsive React dashboard; real API wiring; loading/empty/error/success states; no fake metrics | source **W** (logic + types **T**); never built or rendered |
## Deployment and security
| Dockerfile / Compose / Kubernetes examples / CI | **W** (YAML-parsed only) |
| `.env.example` placeholders; `.gitignore` coverage | **T** |
| Upload safety, path traversal, size limits, input validation, no debug flags, no hard-coded secrets | **T** (service level) + scan |
| Authentication/authorization | **-** (documented) |
## Documentation and delivery
README, architecture, dataset, testing report, security review, Windows setup, deployment, audit report, demo script, reflection template: done. Presentation numbers and reflection: **R** (yours).

## Pass 6 (real-data workflow)
| Requirement | Status |
|---|---|
| Acquire real PaySim | **R / BLOCKED** (no network here); manual steps in `docs/paysim_provenance.md` |
| Extended EDA (balance consistency, amount by class, per-step fraud, single-rule diagnostic, plots) | **T** on synthetic; **R** on real data |
| Leakage-safe features incl. pre-transaction balances; post-transaction set as diagnostic | **T** (feature sets, API/UI aware) |
| Chronological split with documented boundaries; unusable-split handling | **T** (`ml.splits`, training errors) |
| XGBoost training with CPU config, hyperparameter flags, validation-only search, timing/memory | **W** for XGBoost (scikit-learn models **T**) |
| Evaluation: classification report, PR/ROC/confusion/importance (JSON + PNG) | **T** (scikit-learn models) |
| Artifact with schema, metadata, version, metrics; reload and compare | **T** (scikit-learn models); real artifact **R / BLOCKED** |
| API/dashboard use the saved artifact's metrics and curves | **T** service level; HTTP/UI **W** |
| MLflow records dataset reference, config, schema, metrics, duration (no raw data) | **W** |
| Docs: provenance, EDA, features, training, evaluation, reproducibility, model card, Git workflow | done (data-dependent numbers left blank on purpose) |
