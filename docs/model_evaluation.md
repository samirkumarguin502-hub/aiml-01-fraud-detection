# Model evaluation protocol

No real evaluation results exist yet (`docs/real_training_report.md`). This document defines what the code computes and how to read it.

## What is computed (`ml/train.py`, `ml/pipeline/evaluate.py`)
For a logistic-regression baseline and the chosen model (XGBoost by default):
- **Threshold-free:** PR-AUC (average precision) and ROC-AUC; `None` (shown "n/a") when a class is missing, never a made-up number.
- **At the operating threshold:** precision, recall, F1, confusion matrix (TP, FP, FN, TN), flagged share, and a per-class classification report with support.
- **Threshold analysis:** metrics on a grid of thresholds (held-out test) so the trade-off is visible; the chosen threshold is marked.
- **Curves and importance:** precision-recall and ROC points, feature importance (XGBoost gain; |coefficient| for logistic regression), saved in the artifact
  metadata and as PNGs under `ml/artifacts/<run>/plots/` (titles carry the data label, so synthetic results are visibly synthetic).
- **Run facts:** fit seconds, total seconds, peak memory, CPU count, library versions, seed, split boundaries, class balance per split.
Outputs: `report.md` (human-readable), `summary.json` (machine-readable), `plots/`, and the artifact folders `baseline/`, `primary/`.

## Which split is used for what
| Split | Used for | Never used for |
|---|---|---|
| train (earliest steps) | fitting; feature vocabulary; class weights | |
| validation (middle) | early stopping, hyperparameter search (`--search`), threshold, choosing between baseline/primary and between runs | reporting final performance |
| test (latest) | one final report per model at the already-fixed threshold | any choice |
The code enforces both classes in every split (otherwise it stops with an explanatory error) and the search function cannot receive the test split.

## Threshold selection
Highest **validation** recall among thresholds whose validation precision is at least `--min-precision` (default 0.80). If none reaches it, the max-F1
threshold is used and the report says "FALLBACK". Consequence: raising the target gives fewer false alerts (less analyst work) and more missed fraud;
lowering it does the opposite. The target is a business decision, so report the trade-off table, not just one point.

## Class imbalance
Fraud is a tiny fraction of rows, so accuracy would be about 100% for a model that flags nothing and is not reported as a success metric. XGBoost uses
`scale_pos_weight = n_non_fraud / n_fraud` computed from the training split; the baseline uses balanced class weights. No rows are dropped or resampled.
Class weighting makes scores a **ranking score, not a calibrated probability**; the API and dashboard never call it a probability.

## Reading the numbers honestly
- PaySim is simulated; a very high score may reflect the generator's rules rather than difficulty. Run `python -m ml.eda.analyze` and read its single-rule diagnostic,
  and compare feature sets (`python -m ml.compare_runs`). `all_balance` results are diagnostic only.
- One chronological split gives one estimate; fraud counts per split can be small, so do not over-interpret small differences between runs.
- Test numbers belong in the report once per final model; do not iterate on them.

## Verification after saving
Every training run reloads the saved artifact from disk and compares it with the in-memory model on up to 2,000 held-out rows (tolerance 1e-6) and
checks the feature schema; a mismatch aborts the run and removes the folder. Inference needs numpy, pandas and the model library (xgboost or
scikit-learn); it does not need MLflow or matplotlib (tested).
