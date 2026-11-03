# 10-minute demo script (use only what you have actually run)

Rule: every number you say must come from your own screen. Replace each **[bracket]** with a real value or delete the sentence.
If you use the synthetic demo model, say so out loud: the dashboard also shows a SYNTHETIC banner.

Before you start: API running (`docs/windows_setup.md` step 6), dashboard open at http://localhost:5173, a model loaded, a CSV of ~200-2000 rows ready,
`python -m streaming ...` command in a terminal. Have `docs/eda/dataset_inspection.txt` and `ml/artifacts/<run>/report.md` open as backup.

| Min | Segment | What to show / say |
|---|---|---|
| 0:00-1:00 | Problem | Fraud is rare, so accuracy misleads; analysts need a ranked, reviewable alert list. Data: PaySim, **simulated** mobile-money transactions (rows **[N]**, fraud rate **[x]%** from your EDA). |
| 1:00-2:30 | Data and leakage | Show the EDA output. The balance columns reveal the label (fraudulent transactions are cancelled), so they are never loaded: `ml/pipeline/features.py`, `ml/train.py` reads only 6 columns. |
| 2:30-4:00 | Method | Time split by step (train < validation < test); class weights; threshold chosen on validation; test scored once. Show `report.md`: PR-AUC **[..]**, precision/recall at threshold **[..]**, confusion matrix **[..]**. Say what the baseline scored **[..]**. |
| 4:00-5:00 | Architecture | `docs/architecture.md` diagram: file → (replay) → API → model → SQLite → dashboard. Optional parts (Kafka, Feast, Evidently) exist but only say they ran if you ran them. |
| 5:00-6:30 | Live: scoring | Dashboard → *Score one*: submit a transaction, point out "ranking score, not a probability". Then *Batch CSV*: upload, download results. Show an invalid CSV being rejected without echoing values. |
| 6:30-7:30 | Live: analyst workflow | *Alerts & cases*: open an alert, set status, mark fraud/not fraud. *Transactions*: filter and sort. |
| 7:30-8:30 | Live: replay + monitoring | Run `python -m streaming api --data ... --api-mode batch`; watch *Live / replay*. Open *Drift & monitoring*: show the real state (insufficient data or PSI table). Explain that an alert never swaps the model; show the retraining request workflow. |
| 8:30-9:30 | Honest limits | No authentication; simulated data; uncalibrated scores; Kafka/Feast/GNN **[not run / state what you ran]**; history features differ between training and serving. |
| 9:30-10:00 | Close | What you built, what you measured, what you would do next (authentication, real streaming, calibration). |

Likely questions: Why not use balances? (leakage.) Why PR-AUC? (imbalance.) How was the threshold chosen? (validation, precision target.)
Why no test-set tuning? (single held-out report.) What would break in production? (`docs/security_review.md`.)
