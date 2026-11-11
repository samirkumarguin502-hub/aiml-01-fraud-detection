# Real training report

## Status: EXISTING REAL-FILE XGBOOST ARTIFACT VERIFIED FROM METADATA
The workspace contains a user-supplied PaySim-format CSV and the `pre-balance-full-v2` XGBoost artifact. Its provenance label states
`user-supplied file paysim dataset.csv (provenance unverified)`, so external dataset provenance and licence still require independent verification.
Do not use the recorded metrics as proof of a verified PaySim download or a replayable retraining run.

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
| Run id / date | `pre-balance-full-v2` / 2026-11-03T18:04:24Z |
| Dataset file and SHA-256 | `paysim dataset.csv`; dataset SHA-256 `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |
| Feature set and number of features | `pre_balance`; 20 engineered feature names |
| Split boundaries | Train through step 445; validation through step 594; test starts after step 594 |
| Model, seed, hyperparameters | XGBoost; seed 42; 100 estimators; depth 6; learning rate 0.1; best iteration 93; scale_pos_weight 1199.5066906331137 |
| Search used? trials and chosen parameters | |
| Training seconds, peak memory MB, CPUs | |
| Operating threshold and the rule that chose it (validation only) | |
| Validation: PR-AUC, precision, recall, F1 | 0.9966008434643729; 0.9411764705882353; 1.0; 0.9696969696969697 |
| Test (once): PR-AUC, ROC-AUC, precision, recall, F1 | 0.9977122034905092; 0.9999701114127524; 0.9694656488549618; 0.998186215235792; 0.9836163240988979 |
| Test confusion matrix: TP / FP / FN / TN | 1651 / 52 / 3 / 121874 |
| Baseline (logistic regression) test PR-AUC for comparison | |
| Reload check result (max abs difference) | |
| MLflow run id (if used) | |

## Rules this project applies (see `docs/model_evaluation.md`)
Chronological split by `step`; the validation split is used for early stopping, hyperparameter search and the threshold; the test split is scored once
per model after those choices are fixed and is never used to choose between runs. Accuracy is not reported as a success metric. Fraud rows are never
dropped or downsampled; `--nrows` takes the first N rows of the file (the earliest period of a step-sorted file), which changes the time coverage and
must be stated if used.
