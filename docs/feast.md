# Feast (optional)

Status: configuration and consistency logic written and tested **without Feast**; the feature store was **never applied,
materialised or queried** (feast is not installed in the authoring sandbox).

Only entity-keyed history features can come from a feature store here: `orig_prev_txn_count` and `orig_prev_amount_sum`
(entity = sender `nameOrig`; the contract's `orig_prev_amount_mean` = sum / count). Stateless features (amount, type one-hots,
`hour_of_day`, `dest_is_merchant`) come from the request. PaySim has no real timestamps, so the export uses an explicit mapping
`event_timestamp = 2000-01-01 + step hours` (arbitrary epoch, not a date). A row summarising steps ≤ t is stamped t+1, so a
transaction at step s only sees steps < s, exactly like `ml.pipeline.features` (tested by emulating Feast's point-in-time join
with pandas against `build_features`, including same-step exclusion).

Offline (training): `get_historical_features` with the transaction's step timestamp. Online (serving): materialised latest
state = history of all earlier transactions. The API does **not** use Feast; it builds history from its own SQLite rows, and
works identically when Feast is absent.

```
python -m pip install feast pyarrow
python -m ml.feast_export --data ml\data\raw\<file>.csv --out feature_repo\data\orig_history.parquet
cd feature_repo; feast apply
```
No account history is invented: if identifiers or steps are missing the export fails with a clear error.
