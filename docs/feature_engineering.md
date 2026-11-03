# Feature engineering (every feature, by feature set)

Code: `ml/pipeline/features.py` (`FeatureContract`, `build_features`); input rules: `ml/pipeline/schema.py`. One contract is fitted on the
**training range only**, stored in the model artifact, and used unchanged by training, the API and monitoring. Select with
`python -m ml.train --feature-set base|pre_balance|all_balance` (default `base`).

## Never used as features
| Column | Why |
|---|---|
| `isFraud` | the target |
| `isFlaggedFraud` | output of an existing rule system, not an independent input; never even loaded |
| raw `nameOrig`, `nameDest`, `step` | identifiers / raw clock: only derived, non-identifying quantities below are used |

## Feature set `base` (no balance information)
| Feature | Definition | Known at scoring time? |
|---|---|---|
| `amount` | transaction amount | yes |
| `log_amount` | log(1 + amount) | yes |
| `hour_of_day` | step mod 24 | yes |
| `day_index` | step // 24 (excluded from drift monitoring: it grows with time) | yes, but see risk below |
| `dest_is_merchant` | receiver id starts with `M` (uses only the account-type prefix) | yes |
| `orig_prev_txn_count` | earlier transactions by this sender, **strictly earlier steps only** | yes, from history |
| `orig_prev_amount_mean` | mean amount of those transactions (0 if none) | yes |
| `amount_to_prev_mean` | amount / that mean (0 if none) | yes |
| `type__<TYPE>` | one-hot over the types seen in the training range (observed spellings) | yes |
| `type____unknown__` | 1 for a type unseen in training | yes |

## Added by `pre_balance` (balances known BEFORE the transaction executes)
| Feature | Definition |
|---|---|
| `oldbalanceOrg`, `oldbalanceDest` | sender / receiver balance before the transaction |
| `orig_balance_zero`, `dest_balance_zero` | 1 if that balance is 0 |
| `amount_to_orig_balance` | amount / oldbalanceOrg (0 if the balance is 0) |
| `orig_emptied` | 1 if oldbalanceOrg > 0 and amount takes the whole balance |

## Added by `all_balance` (DIAGNOSTIC ONLY: post-transaction information)
| Feature | Definition |
|---|---|
| `newbalanceOrig`, `newbalanceDest` | balances after the transaction |
| `orig_balance_error` | newbalanceOrig + amount - oldbalanceOrg |
| `dest_balance_error` | oldbalanceDest + amount - newbalanceDest |
| `orig_new_zero`, `dest_new_zero` | 1 if that post-transaction balance is 0 |
These are not known when a transaction is scored, and the dataset description says fraud rows are cancelled so they behave differently from
legitimate ones. A model using them measures how optimistic the simulator makes things; it is not a deployable model. The API refuses these inputs
unless the loaded model was trained with this set.

## Which set to use is a data decision, not an assumption
The dataset description warns against all four balance columns; `pre_balance` relies only on the timing argument (they exist before the
transaction). `orig_emptied` may simply restate the simulator's "empty the account, then cash out" rule. Decide with evidence:
1. `python -m ml.eda.analyze` and read the single-rule diagnostic (high precision and recall for one rule = the generator's pattern).
2. Train `base`, `pre_balance` and `all_balance` with different `--run-id`s and compare with `python -m ml.compare_runs` using the **validation** columns.
3. Report the conservative (`base`) result alongside any `pre_balance` result, and describe `all_balance` only as a diagnostic.

## Leakage controls and their tests
No test data in fitting (vocabulary from train only; thresholds from validation); history features use strictly earlier steps (same-step rows are
excluded); altering future rows cannot change earlier features; post-transaction columns and labels change nothing for `base`/`pre_balance`
(`tests/test_pipeline_core.py`, `tests/test_feature_sets.py`).

## Known risks
- `day_index` is a time counter: in a chronological split the test range holds day values never seen in training, so a tree can only treat them as the largest
  training value. If validation shows it helps nothing, drop it (not done here because no real data was available to justify the change).
- Serving history features use only transactions previously scored by the API (train/serve skew; `docs/api.md`).
- Types are matched by exact spelling; unseen spellings go to the unknown bucket.
