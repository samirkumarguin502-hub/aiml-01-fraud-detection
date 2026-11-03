"""Schema validation for PaySim-style CSVs. Fails loudly; never guesses or renames values."""
from __future__ import annotations

import pandas as pd

TARGET = "isFraud"
REQUIRED_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", TARGET]
NUMERIC_COLUMNS = ["step", "amount"]

# Feature sets decide which RAW input columns a model needs (see docs/feature_engineering.md):
#   base         no balance information at all (the original, most conservative contract)
#   pre_balance  adds the two balances known BEFORE the transaction executes (oldbalanceOrg, oldbalanceDest)
#   all_balance  also adds post-transaction balances: DIAGNOSTIC ONLY, label-leaking in PaySim
BASE_INPUTS = ("step", "type", "amount", "nameOrig", "nameDest")
PRE_BALANCE_INPUTS = ("oldbalanceOrg", "oldbalanceDest")
POST_BALANCE_INPUTS = ("newbalanceOrig", "newbalanceDest")
FEATURE_SET_INPUTS = {
    "base": BASE_INPUTS,
    "pre_balance": BASE_INPUTS + PRE_BALANCE_INPUTS,
    "all_balance": BASE_INPUTS + PRE_BALANCE_INPUTS + POST_BALANCE_INPUTS,
}
FEATURE_SETS = tuple(FEATURE_SET_INPUTS)


def feature_set_for_fields(fields) -> str:
    """Smallest feature set whose inputs equal / are covered by `fields` (used by replay tooling)."""
    f = set(fields)
    if set(POST_BALANCE_INPUTS) <= f:
        return "all_balance"
    if set(PRE_BALANCE_INPUTS) <= f:
        return "pre_balance"
    return "base"


class SchemaError(ValueError):
    """Raised when a dataframe does not satisfy the training/serving schema."""


def validate_schema(df: pd.DataFrame, *, require_target: bool = True, feature_set: str = "base") -> dict:
    """Validate columns, dtypes, nulls and target labels. Returns observed facts.

    Transaction-type spellings are reported as observed (e.g. CASH_OUT vs CASH-OUT are
    different strings); they are never normalised here.
    """
    if feature_set not in FEATURE_SET_INPUTS:
        raise SchemaError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    extra = [c for c in FEATURE_SET_INPUTS[feature_set] if c not in BASE_INPUTS]
    required = [c for c in REQUIRED_COLUMNS if require_target or c != TARGET] + extra
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    if len(df) == 0:
        raise SchemaError("Dataframe has no rows.")
    for c in NUMERIC_COLUMNS + extra:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise SchemaError(f"Column '{c}' must be numeric, got {df[c].dtype}.")
    for c in required:
        n = int(df[c].isna().sum())
        if n:
            raise SchemaError(f"Column '{c}' has {n} missing values.")
    if (df["amount"] < 0).any():
        raise SchemaError("Column 'amount' contains negative values.")
    if (df["step"] < 0).any():
        raise SchemaError("Column 'step' contains negative values.")
    for c in extra:
        if (df[c] < 0).any():
            raise SchemaError(f"Column '{c}' contains negative values.")
    if require_target and TARGET in df.columns:
        labels = set(pd.unique(df[TARGET]).tolist())
        if not labels <= {0, 1}:
            raise SchemaError(f"Target '{TARGET}' must be 0/1, found {sorted(labels)}.")
        if len(labels) < 2:
            raise SchemaError(f"Target '{TARGET}' has a single class {sorted(labels)}.")
    return {"rows": int(len(df)), "types_observed": sorted(df["type"].astype(str).unique())}
