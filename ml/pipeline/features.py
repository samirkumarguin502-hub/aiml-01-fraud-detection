"""Shared feature generation for training AND serving.

Leakage policy (see docs/dataset.md): the four balance columns are excluded because the
dataset description states fraudulent transactions are cancelled, so post-transaction
balances encode the label. `isFlaggedFraud` is the output of an existing rule system, not an
input available independently, so it is excluded too. Both exclusions are enforced in code.

Feature availability:
  stateless (online + offline): amount, log_amount, type one-hots, hour_of_day, day_index,
                                dest_is_merchant
  history   (offline batch now; online requires an entity state store, e.g. Feast):
                                orig_prev_txn_count, orig_prev_amount_mean, amount_to_prev_mean
History features use ONLY transactions from strictly earlier steps for the same nameOrig.
Row order inside one step is arbitrary in the data, so same-step rows are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.pipeline.schema import FEATURE_SET_INPUTS, FEATURE_SETS, POST_BALANCE_INPUTS, PRE_BALANCE_INPUTS

LEAKAGE_COLUMNS = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                   "isFlaggedFraud")
FORBIDDEN_FEATURE_COLUMNS = LEAKAGE_COLUMNS + ("isFraud",)
UNKNOWN_TYPE = "__unknown__"
FEATURE_VERSION = "1"

# Extra engineered features per feature set (appended AFTER the base features so base ordering never changes).
PRE_FEATURES = ["oldbalanceOrg", "oldbalanceDest", "orig_balance_zero", "dest_balance_zero",
                "amount_to_orig_balance", "orig_emptied"]
POST_FEATURES = ["newbalanceOrig", "newbalanceDest", "orig_balance_error", "dest_balance_error",
                 "orig_new_zero", "dest_new_zero"]
EXTRA_FEATURES = {"base": [], "pre_balance": PRE_FEATURES, "all_balance": PRE_FEATURES + POST_FEATURES}
# Raw columns each set may legitimately turn into features (everything else in FORBIDDEN stays forbidden).
ALLOWED_RAW = {"base": set(), "pre_balance": set(PRE_BALANCE_INPUTS),
               "all_balance": set(PRE_BALANCE_INPUTS) | set(POST_BALANCE_INPUTS)}


class FeatureContract:
    """Fitted on training data only; serialised with the model so serving matches training."""

    def __init__(self, type_vocab: list[str], feature_set: str = "base"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
        self.feature_set = feature_set
        self.input_columns = list(FEATURE_SET_INPUTS[feature_set])
        self.type_vocab = sorted(type_vocab)
        self.feature_names = (
            ["amount", "log_amount", "hour_of_day", "day_index", "dest_is_merchant",
             "orig_prev_txn_count", "orig_prev_amount_mean", "amount_to_prev_mean"]
            + [f"type__{t}" for t in self.type_vocab] + [f"type__{UNKNOWN_TYPE}"]
            + EXTRA_FEATURES[feature_set]
        )
        forbidden = set(FORBIDDEN_FEATURE_COLUMNS) - ALLOWED_RAW[feature_set]
        bad = set(self.feature_names) & forbidden
        if bad:
            raise ValueError(f"Contract contains forbidden columns: {bad}")

    @classmethod
    def fit(cls, train_df: pd.DataFrame, feature_set: str = "base") -> "FeatureContract":
        return cls(list(train_df["type"].astype(str).unique()), feature_set)

    def to_dict(self) -> dict:
        return {"version": FEATURE_VERSION, "type_vocab": self.type_vocab, "feature_set": self.feature_set,
                "input_columns": self.input_columns, "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureContract":
        if d.get("version") != FEATURE_VERSION:
            raise ValueError(f"Feature contract version {d.get('version')!r} != {FEATURE_VERSION!r}")
        c = cls(d["type_vocab"], d.get("feature_set", "base"))  # artifacts from before feature sets = base
        if "feature_names" in d and list(d["feature_names"]) != c.feature_names:
            raise ValueError("Feature names in the artifact do not match this code version.")
        return c


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-nameOrig count/mean of amounts from strictly earlier steps (causal)."""
    g = (df.groupby(["nameOrig", "step"], sort=True)["amount"]
           .agg(n="count", s="sum").reset_index().sort_values(["nameOrig", "step"]))
    g["cum_n"] = g.groupby("nameOrig")["n"].cumsum() - g["n"]
    g["cum_s"] = g.groupby("nameOrig")["s"].cumsum() - g["s"]
    out = df[["nameOrig", "step"]].merge(g[["nameOrig", "step", "cum_n", "cum_s"]],
                                         on=["nameOrig", "step"], how="left")
    out.index = df.index
    return out


def build_features(df: pd.DataFrame, contract: FeatureContract) -> pd.DataFrame:
    """Return a numeric feature frame in contract order. Never reads leakage columns."""
    for c in contract.input_columns:
        if c not in df.columns:
            raise KeyError(f"build_features requires column '{c}' (feature set '{contract.feature_set}')")
    h = _history_features(df)
    prev_n = h["cum_n"].astype(float)
    prev_mean = np.where(prev_n > 0, h["cum_s"] / prev_n.where(prev_n > 0, 1.0), 0.0)
    X = pd.DataFrame(index=df.index)
    X["amount"] = df["amount"].astype(float)
    X["log_amount"] = np.log1p(X["amount"])
    X["hour_of_day"] = (df["step"] % 24).astype(int)
    X["day_index"] = (df["step"] // 24).astype(int)
    X["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    X["orig_prev_txn_count"] = prev_n
    X["orig_prev_amount_mean"] = prev_mean
    X["amount_to_prev_mean"] = np.where(prev_mean > 0, X["amount"] / np.where(prev_mean > 0, prev_mean, 1), 0.0)
    t = df["type"].astype(str)
    known = t.isin(contract.type_vocab)
    for v in contract.type_vocab:
        X[f"type__{v}"] = (t == v).astype(int)
    X[f"type__{UNKNOWN_TYPE}"] = (~known).astype(int)
    if contract.feature_set != "base":
        oo = df["oldbalanceOrg"].astype(float)
        od = df["oldbalanceDest"].astype(float)
        amt = X["amount"]
        X["oldbalanceOrg"], X["oldbalanceDest"] = oo, od
        X["orig_balance_zero"] = (oo == 0).astype(int)
        X["dest_balance_zero"] = (od == 0).astype(int)
        X["amount_to_orig_balance"] = np.where(oo > 0, amt / np.where(oo > 0, oo, 1.0), 0.0)
        X["orig_emptied"] = ((oo > 0) & (amt >= oo - 0.005)).astype(int)  # amount takes the whole balance
    if contract.feature_set == "all_balance":  # DIAGNOSTIC ONLY: uses post-transaction information
        no = df["newbalanceOrig"].astype(float)
        nd = df["newbalanceDest"].astype(float)
        X["newbalanceOrig"], X["newbalanceDest"] = no, nd
        X["orig_balance_error"] = no + amt - oo
        X["dest_balance_error"] = od + amt - nd
        X["orig_new_zero"] = (no == 0).astype(int)
        X["dest_new_zero"] = (nd == 0).astype(int)
    X = X[contract.feature_names]
    forbidden = set(FORBIDDEN_FEATURE_COLUMNS) - ALLOWED_RAW[contract.feature_set]
    assert not set(X.columns) & forbidden
    return X
