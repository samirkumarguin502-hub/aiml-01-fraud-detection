"""Reference profile + PSI used by monitoring. Pure numpy/pandas.

The profile stores only binned proportions (no raw rows) and is saved inside each model artifact:
  features: from TRAIN features;  score: from the model's VALIDATION scores (train scores are overfit).
`day_index` is excluded from drift because it grows monotonically with time by construction.
PSI thresholds (0.1 / 0.25) are a common rule of thumb, not a statistical test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_VERSION = 1
EXCLUDED_FROM_DRIFT = ("day_index",)
SCORE_EDGES = [round(i / 10, 1) for i in range(1, 10)]  # inner edges -> 10 equal bins on [0,1]
PSI_WARN, PSI_ALERT = 0.10, 0.25
EPS = 1e-4


def _bin_props(values: np.ndarray, edges: list[float]) -> list[float]:
    idx = np.searchsorted(np.asarray(edges, dtype=float), values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return (counts / counts.sum()).tolist() if counts.sum() else [0.0] * (len(edges) + 1)


def _edges_for(values: np.ndarray) -> list[float]:
    u = np.unique(values)
    if len(u) <= 1:
        return []
    if set(u.tolist()) <= {0.0, 1.0}:
        return [0.5]
    return np.unique(np.quantile(values, np.linspace(0, 1, 11)[1:-1])).tolist()


def build_reference_profile(X_train: pd.DataFrame, val_scores) -> dict:
    feats = {}
    for c in X_train.columns:
        if c in EXCLUDED_FROM_DRIFT:
            continue
        v = X_train[c].to_numpy(dtype=float)
        e = _edges_for(v)
        feats[c] = {"edges": e, "props": _bin_props(v, e)}
    s = np.asarray(val_scores, dtype=float)
    return {"version": PROFILE_VERSION, "n_train_rows": int(len(X_train)), "n_validation_scores": int(len(s)),
            "excluded": list(EXCLUDED_FROM_DRIFT), "features": feats,
            "score": {"edges": SCORE_EDGES, "props": _bin_props(s, SCORE_EDGES)}}


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(v: float) -> str:
    return "alert" if v > PSI_ALERT else ("warning" if v > PSI_WARN else "stable")


def psi_against_profile(entry: dict, values) -> float:
    return psi(entry["props"], _bin_props(np.asarray(values, dtype=float), entry["edges"]))
