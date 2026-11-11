"""Schema validation for PaySim-style CSVs.

Fails loudly; never guesses, renames or normalizes source values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "isFraud"

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    TARGET,
]

NUMERIC_COLUMNS = ["step", "amount"]

BASE_INPUTS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
)

PRE_BALANCE_INPUTS = (
    "oldbalanceOrg",
    "oldbalanceDest",
)

POST_BALANCE_INPUTS = (
    "newbalanceOrig",
    "newbalanceDest",
)

FEATURE_SET_INPUTS = {
    "base": BASE_INPUTS,
    "pre_balance": BASE_INPUTS + PRE_BALANCE_INPUTS,
    "all_balance": (
        BASE_INPUTS + PRE_BALANCE_INPUTS + POST_BALANCE_INPUTS
    ),
}

FEATURE_SETS = tuple(FEATURE_SET_INPUTS)


class SchemaError(ValueError):
    """Raised when a dataframe does not satisfy the data schema."""


def feature_set_for_fields(fields) -> str:
    """Return the smallest feature set fully covered by the given fields."""
    available = set(fields)

    for feature_set in ("all_balance", "pre_balance", "base"):
        required = set(FEATURE_SET_INPUTS[feature_set])
        if required.issubset(available):
            return feature_set

    # Base fields are not fully covered. Return the conservative default.
    return "base"


def _validate_numeric_column(
    df: pd.DataFrame,
    column: str,
    *,
    non_negative: bool = False,
    whole_number: bool = False,
) -> None:
    """Validate numeric type, finite values and optional constraints."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise SchemaError(
            f"Column '{column}' must be numeric, got {df[column].dtype}."
        )

    values = df[column].to_numpy(dtype=np.float64, na_value=np.nan)

    if not np.isfinite(values).all():
        raise SchemaError(
            f"Column '{column}' contains NaN or infinite values."
        )

    if non_negative and (values < 0).any():
        raise SchemaError(
            f"Column '{column}' contains negative values."
        )

    if whole_number and not np.equal(values, np.floor(values)).all():
        raise SchemaError(
            f"Column '{column}' must contain whole-number values."
        )


def validate_schema(
    df: pd.DataFrame,
    *,
    require_target: bool = True,
    feature_set: str = "base",
) -> dict:
    """Validate required columns, data types, nulls and target labels.

    Transaction-type spellings are preserved exactly as observed.
    No automatic renaming or normalization is performed.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("Input must be a pandas DataFrame.")

    if feature_set not in FEATURE_SET_INPUTS:
        raise SchemaError(
            f"Unknown feature_set {feature_set!r}; "
            f"choose from {FEATURE_SETS}"
        )

    if df.empty:
        raise SchemaError("Dataframe has no rows.")

    feature_inputs = list(FEATURE_SET_INPUTS[feature_set])
    extra = [
        column
        for column in feature_inputs
        if column not in BASE_INPUTS
    ]

    required = [
        column
        for column in REQUIRED_COLUMNS
        if require_target or column != TARGET
    ] + extra

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise SchemaError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Reject null values in every required column.
    for column in required:
        missing_count = int(df[column].isna().sum())

        if missing_count:
            raise SchemaError(
                f"Column '{column}' has {missing_count} missing values."
            )

    # Numeric input validation.
    _validate_numeric_column(
        df,
        "step",
        non_negative=True,
        whole_number=True,
    )

    _validate_numeric_column(
        df,
        "amount",
        non_negative=True,
    )

    for column in extra:
        _validate_numeric_column(
            df,
            column,
            non_negative=True,
        )

    # Transaction type and account identifiers must not be blank.
    for column in ("type", "nameOrig", "nameDest"):
        values = df[column].astype("string").str.strip()

        if values.eq("").any():
            raise SchemaError(
                f"Column '{column}' contains empty string values."
            )

    # Target validation is required for training/evaluation.
    if require_target:
        _validate_numeric_column(df, TARGET)

        labels = set(pd.unique(df[TARGET]).tolist())

        if not labels.issubset({0, 1}):
            raise SchemaError(
                f"Target '{TARGET}' must contain only 0/1, "
                f"found {sorted(labels)}."
            )

        if len(labels) < 2:
            raise SchemaError(
                f"Target '{TARGET}' has a single class "
                f"{sorted(labels)}."
            )

    return {
        "rows": int(len(df)),
        "types_observed": sorted(
            df["type"].astype(str).unique().tolist()
        ),
        "feature_set": feature_set,
        "target_validated": require_target,
    }