# Reproducibility

## Inputs that must be identical
1. The dataset file: record its **SHA-256** (training does this automatically; it is stored in `summary.json` and the artifact). Do not commit the file.
2. The code version: `git rev-parse HEAD` (commit before a final run so the hash identifies the code).
3. The Python environment: `& $py -m pip freeze > ml\artifacts\<run_id>\requirements-frozen.txt` after training (training also records Python, platform, CPU count and
   the numpy/pandas/scikit-learn/xgboost versions in the artifact metadata). Scikit-learn model files are rejected by a different scikit-learn minor version.
4. The command line: seed (`--seed`, default 42), feature set, split fractions, `--nrows`, hyperparameters, `--n-jobs`.

## What the seed controls
`--seed` seeds Python, NumPy and the model (`random_state`). Splits are deterministic (by `step`, no randomness). Results should repeat exactly on the same
machine, versions and thread count (tested for scikit-learn models). XGBoost with multiple threads can differ slightly across versions or thread counts; the run records `n_jobs` and the
version. Re-run with the same `--n-jobs` to compare.

## Exact steps
```
cd C:\Projects\AIML-01-Fraud-Detection
$py = ".\.venv-test\Scripts\python.exe"
& $py -m ml.eda.analyze
& $py -m ml.splits --data ml\data\raw\<file>.csv
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --feature-set <set> --seed 42 --run-id <id>
```
Add `--search` to select `max_depth`/`learning_rate` on validation PR-AUC; add `--nrows N` for a first trial (state it in any report).

## What a run leaves behind (`ml\artifacts\<run_id>\`, git-ignored)
`report.md`, `summary.json`, `plots\*.png`, and for `baseline\` and `primary\`: the model file (`model.json` for XGBoost, `model.joblib` otherwise), `metadata.json`
(feature contract and schema, threshold and policy, parameters, metrics, curves, reference profile for drift monitoring, training time/memory, library versions,
dataset SHA-256, data label, model version = run id, creation time) and a SHA-256 of the model file that loading verifies.

## MLflow (optional)
If `mlflow` is installed, training logs parameters (dataset SHA-256, data label, feature set, number of features, model parameters), validation and test metrics, fit seconds,
and the run folder as artifacts (models, report, plots; **never the dataset**). Set `$env:MLFLOW_TRACKING_URI` to use a server; otherwise it writes to `mlruns\` (git-ignored).
Without MLflow, training completes and the report says "unavailable". This path has never been executed by the author (MLflow not installed there).
View runs with `& $py -m mlflow ui` (then open the URL it prints).

## Not reproducible from this repository
The dataset (download it yourself), trained artifacts (git-ignored by design), and any MLflow history.
