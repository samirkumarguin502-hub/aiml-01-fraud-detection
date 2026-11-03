# Final audit report: AIML-01 (updated for Pass 6, 30 Oct 2026; deadline 20 Nov 2026)

Details: `docs/testing_report.md`, `docs/security_review.md`, `docs/requirements_checklist.md`.

## 1. Headline (read this first)
**The Pass 6 goal (real PaySim data -> XGBoost -> evaluation -> artifact -> API -> dashboard) was NOT achieved, because it was impossible in the authoring
environment:** there was no network (kaggle.com and PyPI unreachable), no Kaggle credentials, no PaySim file anywhere on disk, and neither XGBoost nor FastAPI/Node
packages were installed. **No real dataset was acquired, no real model was trained, no real metrics exist, and no model artifact is included in the ZIP.** Nothing was
faked to cover this. What Pass 6 did deliver is the machinery and the exact commands to complete the objective on your laptop, verified on synthetic data.

## 2. Demo-readiness: PARTIALLY READY (unchanged in kind, improved in substance)
Before a real demo you must still complete: install dependencies, run real pytest, download PaySim, run EDA/splits, train, start API + UI. Expect small first-contact defects in code that
has never run against real FastAPI, React, XGBoost or PaySim. A rehearsal with labelled synthetic data works as soon as dependencies are installed.

## 3. Added in Pass 6 (verified on synthetic data unless stated)
- **Feature sets** (`--feature-set base|pre_balance|all_balance`): prediction-time-safe balance features, plus a diagnostic set with post-transaction balances; schema, loader, API validation, CSV
  parsing, SQLite (with migration), monitoring, replay client, dashboard forms and model page all follow the loaded model's `input_fields`. Default stays `base` (conservative).
- **Evaluation outputs:** classification report, PR/ROC curve points, feature importance, confusion-matrix/PR/ROC/importance PNGs (titles carry the data label), fit time, peak memory, CPU count.
- **Training controls:** hyperparameter flags, `--search` (validation PR-AUC only; the test split cannot be passed to it), mandatory save-then-reload verification (aborts and cleans up on mismatch).
- **Tools:** `python -m ml.splits` (split sanity check), `python -m ml.compare_runs`, `python -m ml.eda.analyze` (balance consistency, amount/step statistics, single-rule diagnostic, plots, generated `docs/eda_report.md`).
- **Docs:** `paysim_provenance.md` (canonical), `eda_report.md` (placeholder, generated from real data), `feature_engineering.md` (tested against the code), `real_training_report.md`, `model_evaluation.md`,
  `reproducibility.md`, `model_card.md`, `git_workflow.md`, updated `windows_setup.md`.

## 4. Written but never executed (unchanged)
FastAPI HTTP layer; React build/render and component tests; XGBoost; MLflow; Kafka; Feast; Evidently; Docker/Compose/Kubernetes/CI files.

## 5. Not implemented (unchanged)
Authentication/authorization; GNN; live Kafka deployment; cloud deployment; calibration; explanations; hot model reload.

## 6. Dataset and model status
Dataset: not acquired (provenance and terms: `docs/paysim_provenance.md`; licence, date, hash still to be filled by you). Model: none trained on real data. Any artifact made here came from synthetic fixtures
and is labelled SYNTHETIC; none is shipped.

## 7. Open questions only real data can answer
Do `pre_balance` features beat `base` on validation, or do they just restate the simulator's rules (see the single-rule diagnostic)? Is `day_index` helping or hurting under a chronological split?
Does the validation precision target of 0.80 produce a workable alert volume? How much memory does a full-file run need on your laptop?

## 8. Remaining actions for you (priority order)
1. Install and run real tests (`docs/windows_setup.md` §1-2). Send failures verbatim.
2. Download PaySim manually; record licence, date, SHA-256 (`docs/paysim_provenance.md`); run `ml.eda.analyze` and `ml.splits` (§3-5).
3. Short controlled XGBoost trial, then the feature-set comparison, then the final run (§6-8). Fill `docs/real_training_report.md` from the run's own files only.
4. Start API and dashboard with the real artifact; test single/batch prediction (§9-11); check every dashboard page.
5. Replay + monitoring demo, then rehearse `docs/demo_script.md` with real numbers; write your reflection.
6. Commit and push yourself using `docs/git_workflow.md`; confirm no dataset/secret/artifact is staged.
Calendar: about 21 days remain; plan roughly 10-15 hours at 2-3 h/day including first-contact fixes.

## 9. Final ZIP checks
Performed on the staging tree and re-checked on the extracted ZIP: no `.venv*`, `.env`, `*.db`, dataset CSV, model artifacts, `node_modules`, `dist`, caches or `ml/artifacts`; `.env.example` placeholders only;
secret-pattern scan clean; all tests re-run from the extracted copy.
