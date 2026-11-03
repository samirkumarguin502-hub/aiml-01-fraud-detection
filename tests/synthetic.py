"""SYNTHETIC test fixtures only. Numbers produced from this data are NOT real results.

Fraud is planted in TRANSFER/CASH_OUT rows with larger amounts. The four balance columns are
deliberately made to leak the label so leakage tests can prove models never see them.
"""
import numpy as np
import pandas as pd


def make_synthetic(n_steps: int = 60, per_step: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(1, n_steps + 1):
        for i in range(per_step):
            t = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"],
                           p=[0.35, 0.2, 0.2, 0.2, 0.05])
            fraud = int(t in ("TRANSFER", "CASH_OUT") and rng.random() < 0.15)
            amount = float(rng.lognormal(12.0 if fraud else 8.0, 0.6))
            dest = f"M{rng.integers(1, 50)}" if t == "PAYMENT" else f"C{rng.integers(1, 500)}"
            rows.append(dict(step=step, type=t, amount=amount, nameOrig=f"C{step}_{i}",
                             nameDest=dest, oldbalanceOrg=amount * 2,
                             newbalanceOrig=0.0 if fraud else amount,
                             oldbalanceDest=0.0 if fraud else 10.0, newbalanceDest=0.0,
                             isFraud=fraud, isFlaggedFraud=0))
    return pd.DataFrame(rows)
