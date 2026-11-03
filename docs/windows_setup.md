# Windows PowerShell guide (CPU only, existing environment)

Run everything from `C:\Projects\AIML-01-Fraud-Detection`. Calling the venv's `python.exe` directly avoids `Activate.ps1` / execution-policy problems.
`$env:NAME = "..."` lasts for that window only; the app does **not** read `.env` files. Define the shorthand once per window:
```
cd C:\Projects\AIML-01-Fraud-Detection
$py = ".\.venv-test\Scripts\python.exe"
```
Commands that need the real dataset, a trained model, or a missing package are marked **[needs ...]**. Nothing here has been run against the real dataset by the author.

## 1. Check the environment
```
Test-Path .\.venv-test\Scripts\python.exe        # must print True; if False, tell me - do not reinstall Python
& $py --version ; & $py -m pip --version
& $py -m pip list
node --version                                   # >= 22.18 is needed for the frontend logic tests
```

## 2. Install only what is missing
`pip install -r` skips packages that already satisfy the minimum version; it does not upgrade them.
```
& $py -m pip install -r requirements-dev.txt     # pandas, numpy, scikit-learn, joblib, fastapi, uvicorn, pydantic, httpx, pytest, ruff
& $py -m pip install xgboost                     # [needed for --model xgboost]
& $py -m pip install matplotlib                  # [needed for the plots; training still works without it]
& $py -m pip install mlflow                      # OPTIONAL experiment tracking
& $py -m pytest -q                               # runs the synthetic-fixture tests (no dataset needed)
```
If any test fails, paste the output; do not edit the test. (The FastAPI HTTP tests and the real-XGBoost test were skipped, never run, in the authoring environment.)

## 3. Place the real PaySim CSV   [needs the download: docs/paysim_provenance.md]
Put the CSV in `ml\data\raw\` and verify it (do not assume the download worked):
```
Get-Item ml\data\raw\*.csv | Select-Object Name, Length
Get-FileHash -Algorithm SHA256 ml\data\raw\<file>.csv
```
Remove any `SYNTHETIC_*.csv` first: the older EDA scripts use the first `*.csv` in that folder. Record the hash, date and licence in `docs\paysim_provenance.md`.

## 4. Inspect the dataset and generate EDA   [needs the dataset]
```
& $py ml\eda\inspect_dataset.py                  # -> docs\eda\dataset_inspection.txt, dataset_stats.json (no arguments)
& $py ml\eda\make_eda_report.py                  # -> docs\eda\EDA_REPORT.md (no arguments)
& $py -m ml.eda.analyze                          # -> docs\eda_report.md, docs\eda\balance_analysis.json, docs\eda\plots\*.png
                                                 #    options: --data <csv> --nrows N --out-dir <dir> --report <file>
```
Read `docs\eda_report.md`: class balance, fraud by type (note the real spelling, `CASH_OUT` or `CASH-OUT`), balance behaviour by class and the **single-rule diagnostic**.

## 5. Generate and check the chronological split   [needs the dataset]
```
& $py -m ml.splits --data ml\data\raw\<file>.csv           # steps, rows and fraud counts per split; exit code 1 if a split is unusable
& $py -m ml.splits --data ml\data\raw\<file>.csv --json    # same, machine-readable. Options: --nrows --train-frac 0.6 --val-frac 0.2
```

## 6. Train the XGBoost model   [needs the dataset and xgboost]
Always start small and read the report before a full run:
```
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --nrows 500000 --n-estimators 100 --run-id trial1 --no-mlflow
```
Then compare feature sets on the same data (choose using the **validation** columns), and run the full file with your chosen set:
```
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --feature-set base        --run-id fs-base
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --feature-set pre_balance --run-id fs-pre
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --feature-set all_balance --run-id fs-all   # DIAGNOSTIC only
& $py -m ml.compare_runs ml\artifacts\fs-base ml\artifacts\fs-pre ml\artifacts\fs-all
& $py -m ml.train --data ml\data\raw\<file>.csv --model xgboost --feature-set <chosen> --search --run-id final-v1
```
Options (from `ml/train.py`): `--model xgboost|hist_gbm|logreg`, `--feature-set base|pre_balance|all_balance`, `--seed`, `--min-precision 0.80`, `--train-frac`, `--val-frac`, `--nrows`,
`--n-estimators --max-depth --learning-rate --subsample --colsample-bytree --n-jobs --early-stopping-rounds`, `--search`, `--no-plots`, `--run-id`, `--out-dir`, `--data-label`, `--no-mlflow`.
Memory use of a full-file run is **unmeasured** (the report prints peak memory and run time; use a smaller `--nrows` if the laptop struggles). `--nrows` takes the *first* N rows of the file, i.e. the earliest period.

## 7. Evaluate: read the outputs (there is no separate evaluation command)
Training scores the held-out test split once and writes everything to `ml\artifacts\<run_id>\`:
```
Get-Content ml\artifacts\final-v1\report.md        # metrics, confusion matrix, classification report, threshold table, run time/memory
Get-ChildItem ml\artifacts\final-v1\plots          # confusion_matrix.png, precision_recall_curve.png, roc_curve.png, feature_importance.png
```
`summary.json` is the machine-readable version. Fill `docs\real_training_report.md` from these files only.

## 8. Verify the saved artifact (training already reloads and compares it; this repeats it independently)
```
& $py -c "from ml.pipeline.model_io import load_artifact; m = load_artifact(r'ml\artifacts\final-v1\primary'); print(m.metadata['model_version'], m.metadata['feature_set'], m.metadata['model_kind'], m.contract.input_columns, round(m.threshold,6))"
```
The report's "Reload check: passed (max |diff| ...)" line is the in-memory-versus-disk comparison. Loading fails with a clear message if the file was modified, or a library version differs.

## 9. Start FastAPI
```
$env:MODEL_ARTIFACT_DIR = ".\ml\artifacts\final-v1\primary"     # omit to run with no model: predictions return 503 and the dashboard says so
$env:RATE_LIMIT_PREDICT_PER_MIN = "100000" ; $env:RATE_LIMIT_BATCH_PER_MIN = "600"    # only for replay demos
& $py -m uvicorn backend.app.main:create_app --factory --host 127.0.0.1 --port 8000
```
Keep `127.0.0.1`: there is no authentication.

## 10. Start the React frontend   [needs Node >= 22.18 and network for the first npm install]
```
cd frontend
npm install
npm run dev                  # http://localhost:5173 ; /api is proxied to http://127.0.0.1:8000
```

## 11. Test single and batch prediction   (second window; [needs a model])
```
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready          # 503 until a valid model is loaded: intended
(Invoke-RestMethod http://127.0.0.1:8000/api/v1/model).model | Select-Object feature_set, input_fields, model_version
```
`input_fields` tells you exactly what a request must contain. Base feature set:
```
$body = @{ step = 5; type = "TRANSFER"; amount = 1234.5; nameOrig = "C1"; nameDest = "C2" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/predict -ContentType "application/json" -Body $body
```
`pre_balance` models also require `oldbalanceOrg` and `oldbalanceDest` (add them to the hashtable). Use a `type` spelled as in your data.
Batch (columns required by the model must be in the CSV; other columns are ignored):
```
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict/batch -H "Content-Type: text/csv" --data-binary "@ml\data\raw\small_sample.csv"
```
Or use the dashboard pages *Score one* and *Batch CSV*; they adapt to the loaded model's inputs.

## 12. Replay the dataset into the running API and check monitoring   [needs the dataset and a running API]
```
& $py -m streaming api --data ml\data\raw\<file>.csv --nrows 2000 --api-mode batch --batch-size 200
        # --feature-set auto (default) asks the API which fields its model needs; options: --limit --speed --api-url --outcomes-dir
& $py -m monitoring status --db fraud_platform.db --artifact .\ml\artifacts\final-v1\primary
```
Drift numbers need at least 100 scored transactions (`MIN_MONITOR_ROWS`).

## 13. Stop services
`Ctrl+C` in each window. If a window was closed and a port is still busy:
```
Get-NetTCPConnection -LocalPort 8000,5173 -State Listen | Select-Object LocalPort, OwningProcess
Stop-Process -Id <OwningProcess>
```

## No dataset yet? Rehearse with labelled synthetic data (numbers are meaningless)
```
& $py scripts\make_synthetic_demo.py
& $py -m ml.train --data ml\data\raw\SYNTHETIC_demo.csv --model hist_gbm --no-mlflow --run-id demo --data-label "SYNTHETIC demo data"
$env:MODEL_ARTIFACT_DIR = ".\ml\artifacts\demo\primary"
```
The API and dashboard will show SYNTHETIC warnings. Delete `ml\data\raw\SYNTHETIC_demo.csv` before using the real dataset.

## Troubleshooting
| Symptom | Meaning / fix |
|---|---|
| `No module named fastapi` / `xgboost` | step 2 not done in this venv |
| `/ready` returns 503 | no valid model: check `$env:MODEL_ARTIFACT_DIR`; the JSON says why |
| `scikit-learn version mismatch` | artifact trained with another scikit-learn minor version; retrain in this venv |
| `split has 0 fraud rows` | use more rows (`--nrows`) or different `--train-frac/--val-frac`; check `ml.splits` first |
| API returns `validation_error` for a field | the loaded model's `input_fields` differ from your request (base vs balance models) |
| Replay shows `api_429_rate_limited` | raise the rate-limit variables (step 9) or use `--api-mode batch` |
| `ERROR: xgboost is not installed` | `pip install xgboost`, or use `--model hist_gbm` knowingly |
| Plots missing | install `matplotlib` |
