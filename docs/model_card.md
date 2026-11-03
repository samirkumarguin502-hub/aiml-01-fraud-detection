# Model card: AIML-01 fraud scoring model

**Status: no real model has been trained.** Fields marked *(to fill)* come from your first real run (`docs/real_training_report.md`).

| | |
|---|---|
| Purpose | Rank simulated mobile-money transactions by fraud-likelihood so an analyst can review the highest-ranked ones. A teaching/capstone system, not a production control. |
| Model | XGBoost gradient-boosted trees (`hist` tree method, CPU), with a logistic-regression baseline; scikit-learn gradient boosting is available as a dependency-light fallback. |
| Training data | PaySim simulated transactions (`docs/paysim_provenance.md`): **not real customer data**. Rows, fraud rate: *(to fill)* |
| Inputs | Depends on the feature set (`docs/feature_engineering.md`): `base` = type, amount, time-of-day, receiver-is-merchant, sender history; `pre_balance` adds balances known before the transaction. The loaded model's inputs are reported by `GET /api/v1/model` (`input_fields`). |
| Output | A **ranking score** (0-1) and a flag when it reaches the operating threshold. **Not a calibrated probability.** |
| Operating threshold | Chosen on validation data to maximise recall subject to precision ≥ 0.80 (configurable). Value and consequences: *(to fill)* |
| Evaluation | Chronological split, test scored once. Metrics: *(to fill)*; see `docs/model_evaluation.md`. |
| Version / artifact | `model_version` = run id; artifact hash recorded; stored outside Git (`docs/git_workflow.md`). |

## Intended use
Demonstration and study of imbalance-aware, leakage-aware fraud modelling; analyst-review workflow; drift monitoring concepts.

## Out of scope / do not use for
Decisions about real people or accounts; any claim of real-world accuracy; automated blocking; use without human review.

## Limitations and risks
- Simulated data and simulator-specific patterns; performance will not transfer.
- Balance features may encode the generator's rules; `all_balance` is diagnostic only (uses post-transaction information).
- A single time split; few positives per split can make estimates noisy.
- Scores are uncalibrated; class weighting distorts probabilities.
- Sender-history features at serving time use only transactions this platform has seen (train/serve skew).
- Unseen transaction types are scored in an "unknown" bucket with no learned behaviour.
- No fairness analysis is possible: PaySim contains no demographic or customer attributes.
- No authentication on the serving API; not production-secure (`docs/security_review.md`).

## Monitoring and updates
PSI drift against a reference profile saved with the artifact; retraining is a manual, audited workflow with a server-side gate and no automatic promotion (`docs/monitoring.md`).
