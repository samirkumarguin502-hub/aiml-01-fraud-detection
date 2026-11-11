"""Shared feature generation for training AND serving.

Leakage policy:
- Post-transaction balance columns are excluded from production feature sets.
- isFlaggedFraud and isFraud are never model inputs.
- all_balance is diagnostic-only and must not be used for production inference.

History features use ONLY transactions from strictly earlier steps for the
same nameOrig. Transactions within the same step never influence one another.

Online serving note:
History features computed from a dataframe are batch-history features.
True online serving requires persistent per-account state, such as a Feast
online store or another entity-state store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.pipeline.schema import (
    FEATURE_SET_INPUTS,
    FEATURE_SETS,
    POST_BALANCE_INPUTS,
    PRE_BALANCE_INPUTS,
)

LEAKAGE_COLUMNS = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)

FORBIDDEN_FEATURE_COLUMNS = LEAKAGE_COLUMNS + ("isFraud",)

UNKNOWN_TYPE = "__unknown__"
FEATURE_VERSION = "1"

PRE_FEATURES = [
    "oldbalanceOrg",
    "oldbalanceDest",
    "orig_balance_zero",
    "dest_balance_zero",
    "amount_to_orig_balance",
    "orig_emptied",
]

POST_FEATURES = [
    "newbalanceOrig",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "orig_new_zero",
    "dest_new_zero",
]

EXTRA_FEATURES = {
    "base": [],
    "pre_balance": PRE_FEATURES,
    "all_balance": PRE_FEATURES + POST_FEATURES,
}

ALLOWED_RAW = {
    "base": set(),
    "pre_balance": set(PRE_BALANCE_INPUTS),
    "all_balance": set(PRE_BALANCE_INPUTS) | set(POST_BALANCE_INPUTS),
}

BASE_FEATURES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "day_index",
    "dest_is_merchant",
    "orig_prev_txn_count",
    "orig_prev_amount_mean",
    "amount_to_prev_mean",
]


class FeatureContract:
    """Fitted on training data only and serialized with the model."""

    def __init__(
        self,
        type_vocab: list[str],
        feature_set: str = "base",
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set {feature_set!r}; "
                f"choose from {FEATURE_SETS}"
            )

        if not isinstance(type_vocab, (list, tuple)):
            raise ValueError("type_vocab must be a list or tuple.")

        normalized_vocab = [
            str(value) for value in type_vocab
        ]

        if len(normalized_vocab) != len(set(normalized_vocab)):
            raise ValueError("type_vocab contains duplicate values.")

        if UNKNOWN_TYPE in normalized_vocab:
            raise ValueError(
                f"{UNKNOWN_TYPE!r} is reserved for unknown transaction types."
            )

        self.feature_set = feature_set
        self.input_columns = list(FEATURE_SET_INPUTS[feature_set])
        self.type_vocab = sorted(normalized_vocab)

        self.feature_names = (
            BASE_FEATURES.copy()
            + [f"type__{value}" for value in self.type_vocab]
            + [f"type__{UNKNOWN_TYPE}"]
            + EXTRA_FEATURES[feature_set]
        )

        forbidden = (
            set(FORBIDDEN_FEATURE_COLUMNS)
            - ALLOWED_RAW[feature_set]
        )

        bad = set(self.feature_names) & forbidden

        if bad:
            raise ValueError(
                f"Contract contains forbidden columns: {sorted(bad)}"
            )

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Feature contract contains duplicate feature names.")

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        feature_set: str = "base",
    ) -> "FeatureContract":
        if "type" not in train_df.columns:
            raise KeyError("FeatureContract.fit requires column 'type'.")

        if train_df.empty:
            raise ValueError("Cannot fit a feature contract on an empty dataset.")

        return cls(
            list(train_df["type"].astype(str).unique()),
            feature_set,
        )

    def to_dict(self) -> dict:
        return {
            "version": FEATURE_VERSION,
            "type_vocab": self.type_vocab,
            "feature_set": self.feature_set,
            "input_columns": self.input_columns,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureContract":
        if not isinstance(data, dict):
            raise ValueError("Feature contract must be a dictionary.")

        if data.get("version") != FEATURE_VERSION:
            raise ValueError(
                f"Feature contract version {data.get('version')!r} "
                f"!= {FEATURE_VERSION!r}"
            )

        contract = cls(
            data["type_vocab"],
            data.get("feature_set", "base"),
        )

        if (
            "input_columns" in data
            and list(data["input_columns"]) != contract.input_columns
        ):
            raise ValueError(
                "Input columns in the artifact do not match this code version."
            )

        if (
            "feature_names" in data
            and list(data["feature_names"]) != contract.feature_names
        ):
            raise ValueError(
                "Feature names in the artifact do not match this code version."
            )

        return contract


def _validate_feature_input(
    df: pd.DataFrame,
    contract: FeatureContract,
) -> None:
    """Validate required raw columns and numeric inputs."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Feature input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot build features from an empty dataframe.")

    missing = [
        column
        for column in contract.input_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required feature inputs for "
            f"'{contract.feature_set}': {missing}"
        )

    common_required = [
        "nameOrig",
        "nameDest",
        "step",
        "amount",
        "type",
    ]

    missing_common = [
        column for column in common_required
        if column not in df.columns
    ]

    if missing_common:
        raise KeyError(
            f"Missing required columns: {missing_common}"
        )

    for column in ("nameOrig", "nameDest", "type", "step", "amount"):
        if df[column].isna().any():
            raise ValueError(
                f"Null values found in required column '{column}'."
            )

    for column in ("step", "amount"):
        values = pd.to_numeric(df[column], errors="coerce")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Column '{column}' contains non-finite or non-numeric values."
            )

    if (pd.to_numeric(df["step"]) < 0).any():
        raise ValueError("Transaction steps cannot be negative.")

    if (pd.to_numeric(df["amount"]) < 0).any():
        raise ValueError("Transaction amounts cannot be negative.")

    if contract.feature_set in ("pre_balance", "all_balance"):
        for column in PRE_BALANCE_INPUTS:
            values = pd.to_numeric(df[column], errors="coerce")

            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Invalid numeric values in '{column}'."
                )

            if (values < 0).any():
                raise ValueError(
                    f"Negative balance values found in '{column}'."
                )

    if contract.feature_set == "all_balance":
        for column in POST_BALANCE_INPUTS:
            values = pd.to_numeric(df[column], errors="coerce")

            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Invalid numeric values in '{column}'."
                )


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sender history from strictly earlier transaction steps.

    The implementation uses a temporary positional row identifier. This
    preserves the original row order and works even when the input dataframe
    has duplicate index labels.
    """

    required = ("nameOrig", "step", "amount")

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"History calculation requires columns: {missing}"
        )

    if df.empty:
        return pd.DataFrame(
            {
                "cum_n": pd.Series(dtype=float),
                "cum_s": pd.Series(dtype=float),
            },
            index=df.index,
        )

    work = pd.DataFrame({
        "_row_id": np.arange(len(df), dtype=np.int64),
        "nameOrig": df["nameOrig"].astype(str).to_numpy(),
        "step": df["step"].to_numpy(),
        "amount": df["amount"].to_numpy(dtype=float),
    })

    # Aggregate all transactions for the same sender and step.
    # This ensures arbitrary row ordering within one step cannot leak
    # information between transactions.
    grouped = (
        work.groupby(
            ["nameOrig", "step"],
            sort=True,
            as_index=False,
        )
        .agg(
            transaction_count=("amount", "count"),
            transaction_sum=("amount", "sum"),
        )
        .sort_values(
            ["nameOrig", "step"],
            kind="mergesort",
        )
    )

    grouped["cum_n"] = (
        grouped.groupby("nameOrig", sort=False)["transaction_count"]
        .cumsum()
        - grouped["transaction_count"]
    )

    grouped["cum_s"] = (
        grouped.groupby("nameOrig", sort=False)["transaction_sum"]
        .cumsum()
        - grouped["transaction_sum"]
    )

    # Attach the history available at each sender-step to every original row.
    row_history = work[["_row_id", "nameOrig", "step"]].merge(
        grouped[
            ["nameOrig", "step", "cum_n", "cum_s"]
        ],
        on=["nameOrig", "step"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    row_history = row_history.sort_values(
        "_row_id",
        kind="mergesort",
    )

    return pd.DataFrame(
        {
            "cum_n": row_history["cum_n"].to_numpy(dtype=float),
            "cum_s": row_history["cum_s"].to_numpy(dtype=float),
        },
        index=df.index,
    )


def build_features(
    df: pd.DataFrame,
    contract: FeatureContract,
) -> pd.DataFrame:
    """Return numeric features in the exact fitted contract order.

    Never reads forbidden target or post-transaction leakage columns for
    production feature sets.
    """

    _validate_feature_input(df, contract)

    history = _history_features(df)

    previous_count = history["cum_n"].astype(float)
    previous_sum = history["cum_s"].astype(float)

    previous_mean = np.divide(
        previous_sum.to_numpy(dtype=float),
        previous_count.to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=previous_count.to_numpy(dtype=float) > 0,
    )

    amount = pd.to_numeric(
        df["amount"],
        errors="raise",
    ).astype(float)

    X = pd.DataFrame(index=df.index)

    X["amount"] = amount
    X["log_amount"] = np.log1p(amount)

    X["hour_of_day"] = (
        pd.to_numeric(df["step"]) % 24
    ).astype(int)

    X["day_index"] = (
        pd.to_numeric(df["step"]) // 24
    ).astype(int)

    X["dest_is_merchant"] = (
        df["nameDest"]
        .astype(str)
        .str.startswith("M")
        .astype(int)
    )

    X["orig_prev_txn_count"] = previous_count.to_numpy()
    X["orig_prev_amount_mean"] = previous_mean

    X["amount_to_prev_mean"] = np.divide(
        amount.to_numpy(dtype=float),
        previous_mean,
        out=np.zeros(len(df), dtype=float),
        where=previous_mean > 0,
    )

    transaction_type = df["type"].astype(str)
    known_types = transaction_type.isin(contract.type_vocab)

    for value in contract.type_vocab:
        X[f"type__{value}"] = (
            transaction_type == value
        ).astype(int)

    X[f"type__{UNKNOWN_TYPE}"] = (
        ~known_types
    ).astype(int)

    # Pre-transaction balances are available only in the corresponding
    # explicitly selected feature set.
    if contract.feature_set in ("pre_balance", "all_balance"):
        old_orig = df["oldbalanceOrg"].astype(float)
        old_dest = df["oldbalanceDest"].astype(float)

        X["oldbalanceOrg"] = old_orig
        X["oldbalanceDest"] = old_dest

        X["orig_balance_zero"] = (
            old_orig == 0
        ).astype(int)

        X["dest_balance_zero"] = (
            old_dest == 0
        ).astype(int)

        X["amount_to_orig_balance"] = np.divide(
            amount.to_numpy(dtype=float),
            old_orig.to_numpy(dtype=float),
            out=np.zeros(len(df), dtype=float),
            where=old_orig.to_numpy(dtype=float) > 0,
        )

        X["orig_emptied"] = (
            (old_orig > 0)
            & (amount >= old_orig - 0.005)
        ).astype(int)

    # Diagnostic-only feature set. These columns contain post-transaction
    # information and must not be used in production fraud prediction.
    if contract.feature_set == "all_balance":
        new_orig = df["newbalanceOrig"].astype(float)
        new_dest = df["newbalanceDest"].astype(float)

        X["newbalanceOrig"] = new_orig
        X["newbalanceDest"] = new_dest

        X["orig_balance_error"] = (
            new_orig + amount - old_orig
        )

        X["dest_balance_error"] = (
            old_dest + amount - new_dest
        )

        X["orig_new_zero"] = (
            new_orig == 0
        ).astype(int)

        X["dest_new_zero"] = (
            new_dest == 0
        ).astype(int)

    # Enforce exact model input order.
    missing_features = [
        name for name in contract.feature_names
        if name not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Generated features missing from contract: {missing_features}"
        )

    X = X.loc[:, contract.feature_names].copy()

    forbidden = (
        set(FORBIDDEN_FEATURE_COLUMNS)
        - ALLOWED_RAW[contract.feature_set]
    )

    leaked = set(X.columns) & forbidden

    if leaked:
        raise ValueError(
            f"Forbidden feature columns detected: {sorted(leaked)}"
        )

    # Final model-input integrity check.
    numeric_values = X.to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Generated feature matrix contains NaN or infinite values."
        )

    return X