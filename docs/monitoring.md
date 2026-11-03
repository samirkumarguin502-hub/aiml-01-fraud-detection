# Monitoring and controlled retraining

Computed by `monitoring/drift.py` from transactions stored in SQLite, against a **reference profile saved inside each model
artifact** (binned train-feature proportions and validation-score proportions; no raw rows). Older artifacts without a
profile report `no_reference`.

| State | Meaning |
|---|---|
| `no_model` | no usable model: only volume is shown |
| `no_reference` | artifact predates monitoring: retrain to create a profile |
| `insufficient_data` | fewer than `MIN_MONITOR_ROWS` (default 100) scored rows: **no drift numbers are produced** |
| `ok` | PSI per feature + score distribution; levels stable < 0.10 ≤ warning < 0.25 ≤ alert (rule of thumb, not a test) |

Also reported: rejected requests by error code (content never stored), unknown-type rate, volume/alert rate, and
precision/recall on **analyst-labelled** transactions (needs ≥ 20 labels with both classes; labelled sample is biased and
labelled as such). `day_index` is excluded from drift because it increases with time by construction.

## Evidently
Optional HTML report: `python -m monitoring evidently --data <csv> --artifact <dir> --db <sqlite> --out report.html`.
**Not verified**: Evidently is not installed in the authoring sandbox and its API changed between releases; the adapter tries
the new API then the legacy one and returns a status instead of failing. The platform's own PSI monitoring does not need it.
`python -m monitoring status --db <sqlite> --artifact <dir>` prints the API's report offline.

## Retraining is a workflow, never an automatic action
A drift alert changes nothing. A human may create a request (`POST /monitoring/retraining`); states are
`requested → candidate_trained → evaluated_passed|evaluated_failed → promoted|rejected` (or `cancelled`), enforced in SQLite
with an audit trail. The *server* runs the evaluation gate (candidate artifact must load and its held-out test PR-AUC must be ≥ the
active model's); a client cannot assert a pass. **"Promoted" records a decision only**: to activate a model, point
`MODEL_ARTIFACT_DIR` at it and restart. Caveats: the two models were evaluated on their own held-out windows, so the gate is a
screen, not proof; there is no authentication yet, so anyone with network access could advance a request.

## Pass 6 note
When the active model needs balance inputs, drift is computed only on stored rows that have them; rows scored earlier by a model without balances are skipped and `rows_compared` says how many were used.
