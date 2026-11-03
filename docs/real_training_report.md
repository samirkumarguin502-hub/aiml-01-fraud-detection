# Real training report

## Status: NOT TRAINED ON REAL DATA (BLOCKED)
No real PaySim run exists. The authoring environment had no network (the dataset could not be obtained), and neither XGBoost nor MLflow was installed
there. Everything below is a **template to complete from your own run**; no number in this file is a result. All training code was exercised only
on synthetic fixtures (`docs/testing_report.md`), where the planted pattern makes every model look perfect.

## What to run (Windows; details in `docs/windows_setup.md`)
1. `& $py -m ml.eda.analyze` then `& $py -m ml.splits --data <csv>` (all splits must be usable).
2. A short, controlled baseline first: `& $py -m ml.train --data <csv> --nrows 500000 --model xgboost --n-estimators 100 --run-id trial1`
3. Feature-set comparison on the same data: `--feature-set base`, `pre_balance`, `all_balance` with distinct `--run-id`s; compare with
   `& $py -m ml.compare_runs ml\artifacts\<runA> ml\artifacts\<runB> ...` (choose on the validation column).
4. Full run with the chosen feature set (optionally `--search`), then copy results into this file from `ml\artifacts\<run_id>\report.md` and `summary.json`.
Do not start the full run before step 2 has finished and you have read its report; its run time and peak memory are printed in the report.

## Fill-in record (copy exactly from `report.md` / `summary.json`; leave blank until you have a run)
| Item | Value |
|---|---|
| Run id / date | |
| Dataset file, size, SHA-256, rows used (`--nrows`?) | |
| Feature set and number of features | |
| Split boundaries (steps) and rows: train / validation / test | |
| Fraud rows and rate per split | |
| Model, seed, hyperparameters (incl. `scale_pos_weight`, best iteration) | |
| Search used? trials and chosen parameters | |
| Training seconds, peak memory MB, CPUs | |
| Operating threshold and the rule that chose it (validation only) | |
| Validation: PR-AUC, precision, recall, F1 | |
| Test (once): PR-AUC, ROC-AUC, precision, recall, F1 | |
| Test confusion matrix: TP / FP / FN / TN | |
| Baseline (logistic regression) test PR-AUC for comparison | |
| Reload check result (max abs difference) | |
| MLflow run id (if used) | |

## Rules this project applies (see `docs/model_evaluation.md`)
Chronological split by `step`; the validation split is used for early stopping, hyperparameter search and the threshold; the test split is scored once
per model after those choices are fixed and is never used to choose between runs. Accuracy is not reported as a success metric. Fraud rows are never
dropped or downsampled; `--nrows` takes the first N rows of the file (the earliest period of a step-sorted file), which changes the time coverage and
must be stated if used.
