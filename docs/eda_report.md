# EDA report (NOT GENERATED YET)

This file is **overwritten** by `python -m ml.eda.analyze` with numbers computed from the real PaySim CSV. No dataset was available when this
project was packaged, so there are no results here and none are invented.

```
& $py -m ml.eda.analyze          # uses the first *.csv in ml\data\raw\ (or --data <csv> --nrows N)
```
It also writes `docs/eda/balance_analysis.json` and plots to `docs/eda/plots/`. The report covers: rows/columns/dtypes, missing values, duplicates,
class balance, fraud by transaction type (observed spellings), amount statistics by class, time-step distribution, balance behaviour by class,
and a single-rule diagnostic that shows how much of the fraud one simple pre-transaction rule explains (the direct test of whether balance fields
reveal the generator's rules). The older `ml/eda/inspect_dataset.py` and `make_eda_report.py` outputs remain available in `docs/eda/`.
