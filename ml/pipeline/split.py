"""Time-aware split on PaySim's step column (1 step = 1 hour).

All rows from a given step remain in the same split. No random row-level
splitting is performed.
"""
from __future__ import annotations

import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
):
    """Return chronological train, validation and test DataFrames.

    Guarantees:
        max(train.step) < min(validation.step)
        max(validation.step) < min(test.step)

    Fractions apply to the number of unique time steps, not row counts.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "step" not in df.columns:
        raise ValueError("Dataset must contain a 'step' column")

    if df.empty:
        raise ValueError("Cannot split an empty dataset")

    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be between 0 and 1")

    if not (0 < val_frac < 1):
        raise ValueError("val_frac must be between 0 and 1")

    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")

    step_values = pd.to_numeric(df["step"], errors="coerce")

    if step_values.isna().any():
        raise ValueError("'step' contains missing or non-numeric values")

    if not step_values.map(lambda value: float(value).is_integer()).all():
        raise ValueError("'step' must contain whole-number time steps")

    steps = sorted(step_values.unique().tolist())

    if len(steps) < 3:
        raise ValueError(
            f"Need at least 3 distinct steps to split, found {len(steps)}"
        )

    n_steps = len(steps)

    train_end = max(1, int(n_steps * train_frac))
    val_end = max(
        train_end + 1,
        int(n_steps * (train_frac + val_frac)),
    )
    val_end = min(val_end, n_steps - 1)

    train_max_step = steps[train_end - 1]
    val_max_step = steps[val_end - 1]

    train = df.loc[step_values <= train_max_step].copy()
    val = df.loc[
        (step_values > train_max_step)
        & (step_values <= val_max_step)
    ].copy()
    test = df.loc[step_values > val_max_step].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("A chronological split is empty; adjust fractions")

    # Explicit temporal-order assertions: prevent accidental leakage.
    if not (
        train["step"].max()
        < val["step"].min()
        and val["step"].max()
        < test["step"].min()
    ):
        raise RuntimeError("Chronological split ordering validation failed")

    boundaries = {
        "train_max_step": int(train_max_step),
        "val_max_step": int(val_max_step),
    }

    return train, val, test, boundaries