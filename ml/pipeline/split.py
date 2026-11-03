"""Time-aware split on the PaySim `step` column (1 step = 1 hour). Whole steps only."""
from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Split by unique step so every train step < every val step < every test step."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1")
    steps = sorted(df["step"].unique())
    if len(steps) < 3:
        raise ValueError(f"Need at least 3 distinct steps to split, found {len(steps)}")
    i_tr = max(1, int(len(steps) * train_frac))
    i_va = max(i_tr + 1, int(len(steps) * (train_frac + val_frac)))
    i_va = min(i_va, len(steps) - 1)
    tr_max, va_max = steps[i_tr - 1], steps[i_va - 1]
    train = df[df["step"] <= tr_max]
    val = df[(df["step"] > tr_max) & (df["step"] <= va_max)]
    test = df[df["step"] > va_max]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty; adjust fractions.")
    return train, val, test, {"train_max_step": int(tr_max), "val_max_step": int(va_max)}
