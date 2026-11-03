# Model limitations and known risks

Status: **no real-data model has been trained yet.** Every number produced so far (in tests) came from
synthetic fixtures and says nothing about PaySim or real fraud.

## Protocol (what the code enforces)
- Time split on `step` (whole steps): train < validation < test. Test is scored once per model, after the
  threshold is fixed on validation. Model choice uses validation PR-AUC only.
- Only `step, type, amount, nameOrig, nameDest, isFraud` are read from the CSV. The four balance columns and
  `isFlaggedFraud` are never loaded, so they cannot reach the model.
- History features (`orig_prev_*`) use strictly earlier steps. In PaySim most `nameOrig` values may appear once,
  so these features may be near-constant; check real EDA before claiming they help.
- Imbalance: `scale_pos_weight = n_neg/n_pos` (XGBoost) or `class_weight="balanced"` (baseline, hist_gbm),
  computed from train labels only.
- Threshold policy: maximise validation recall subject to validation precision >= `--min-precision`
  (default 0.80); if impossible, max F1 (flagged in the report). Raising it means fewer analyst alerts and more
  missed fraud.

## Known limitations
- Scores are ranking scores from class-weighted models, **not calibrated probabilities**. Do not show them as
  "% chance of fraud" in the UI without a calibration test.
- PaySim is simulated. Good results show the model learns the simulator's patterns, not real-world performance.
  The simulator's fraud pattern (take over account, transfer, cash out) may make some features unrealistically
  easy.
- Online serving of history features needs per-account state (Feast or similar). Until that exists, serving
  scores new transactions with the same code path but only the history present in the supplied frame.
- A single time split gives one estimate; fraud rate per period may differ. No confidence intervals yet.
- Unseen transaction types go to an `__unknown__` bucket; the model has no learned behaviour for them.
- Artifacts: `model.json` (XGBoost native) is safe to parse; `model.joblib` is pickle-based. Load only artifacts
  you trained yourself. Loading checks a SHA-256 and library versions, which detects corruption/tampering but
  is not a substitute for trust in the source.
- XGBoost and MLflow code paths are written against their documented APIs but were **not executed** in the
  authoring sandbox (not installed). The first real run is the real test.
