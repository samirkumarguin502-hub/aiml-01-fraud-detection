"""Feast integration helpers that work WITHOUT Feast installed.

What Feast can serve here: only entity-keyed history features (entity = sender `nameOrig`):
orig_prev_txn_count and orig_prev_amount_sum (the contract's orig_prev_amount_mean = sum/count).
Stateless features (amount, type one-hots, hour_of_day, dest_is_merchant) come from the request itself.
PaySim has NO real timestamps, only `step`. Feast needs timestamps, so we use an explicit, documented
mapping  event_timestamp = 2000-01-01 + step hours  (arbitrary epoch; NOT a real date).

Point-in-time semantics: a feature row summarising all transactions with step <= t is stamped at
(t + 1) hours. A transaction at step s is joined to the latest row stamped <= s, i.e. the state after
step s-1, so only strictly earlier steps are visible: identical to ml.pipeline.features.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = datetime(2000, 1, 1)
FEAST_FEATURES = ("orig_prev_txn_count", "orig_prev_amount_sum")


class FeastSupportError(RuntimeError):
    pass


def feast_status() -> dict:
    ok = importlib.util.find_spec("feast") is not None
    return {"library_installed": ok, "note": "Installed only means importable; a feature store was not applied or verified."}


def step_to_timestamp(step: int) -> datetime:
    return EPOCH + timedelta(hours=int(step))


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (nameOrig, step) cumulative state AFTER that step, stamped at step+1 hours."""
    for c in ("nameOrig", "step", "amount"):
        if c not in df.columns:
            raise FeastSupportError(f"Missing column '{c}'")
    g = (df.groupby(["nameOrig", "step"], sort=True)["amount"].agg(n="count", s="sum").reset_index()
           .sort_values(["nameOrig", "step"]))
    out = pd.DataFrame({
        "nameOrig": g["nameOrig"].to_numpy(),
        "feature_step": (g["step"].to_numpy() + 1).astype("int64"),
        "orig_prev_txn_count": g.groupby("nameOrig")["n"].cumsum().to_numpy(dtype=float),
        "orig_prev_amount_sum": g.groupby("nameOrig")["s"].cumsum().to_numpy(dtype=float),
    })
    out["event_timestamp"] = [step_to_timestamp(s) for s in out["feature_step"]]
    return out


def point_in_time_lookup(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Emulates Feast get_historical_features (latest row stamped <= query time) with pandas, and
    derives the contract's history features. Used to test consistency without Feast."""
    q = df[["nameOrig", "step"]].copy()
    q["_i"] = np.arange(len(q))
    q = q.sort_values("step", kind="stable")
    t = table.sort_values("feature_step", kind="stable")[["nameOrig", "feature_step", *FEAST_FEATURES]]
    m = pd.merge_asof(q, t, left_on="step", right_on="feature_step", by="nameOrig",
                      direction="backward", allow_exact_matches=True).sort_values("_i")
    cnt = m["orig_prev_txn_count"].fillna(0.0).to_numpy()
    sm = m["orig_prev_amount_sum"].fillna(0.0).to_numpy()
    mean = np.where(cnt > 0, sm / np.where(cnt > 0, cnt, 1.0), 0.0)
    return pd.DataFrame({"orig_prev_txn_count": cnt, "orig_prev_amount_mean": mean}, index=df.index)


def write_feature_table(df: pd.DataFrame, out_path: Path) -> Path:
    if importlib.util.find_spec("pyarrow") is None:
        raise FeastSupportError("pyarrow is required to write parquet: python -m pip install pyarrow")
    table = build_feature_table(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table[["nameOrig", "event_timestamp", *FEAST_FEATURES]].to_parquet(out_path, index=False)
    return out_path
