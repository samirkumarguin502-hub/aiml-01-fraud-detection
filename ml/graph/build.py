"""Account-transfer graph from the ONLY relationship PaySim contains: sender (nameOrig) -> receiver (nameDest).

No device, card, IP or merchant-attribute identifiers exist in the supported schema, so no
account-device-card graph is built and no fraud rings are asserted. Identifier columns of that kind,
if a different dataset has them, are reported as present-but-unused rather than silently used.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from ml.pipeline.evaluate import ranking_metrics
from ml.pipeline.schema import TARGET
from ml.pipeline.split import time_split

REQUIRED = ("nameOrig", "nameDest", "step")
UNSUPPORTED_IDENTIFIER_HINTS = ("deviceId", "device_id", "cardId", "card_id", "ipAddress", "ip_address")


class GraphInputError(ValueError):
    pass


def graph_status() -> dict:
    def has(m):
        try:
            return importlib.util.find_spec(m) is not None
        except (ImportError, ValueError):
            return False
    return {"torch_installed": has("torch"), "torch_geometric_installed": has("torch_geometric"),
            "gnn_training": "not implemented in this project; only graph construction, structural analysis "
                            "and a leak-free degree baseline exist",
            "core_demo_requires_graph": False}


def validate_graph_input(df: pd.DataFrame) -> dict:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise GraphInputError(f"Missing columns for graph analysis: {missing}")
    if len(df) == 0:
        raise GraphInputError("No rows.")
    if df[list(REQUIRED)].isna().any().any():
        raise GraphInputError("Null identifiers/steps present.")
    return {"identifier_columns_used": ["nameOrig", "nameDest"],
            "unsupported_identifier_columns_present_but_unused": [c for c in UNSUPPORTED_IDENTIFIER_HINTS if c in df.columns],
            "device_card_relationships": "not available in the supported schema; none are constructed"}


def _components(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def graph_summary(df: pd.DataFrame) -> dict:
    facts = validate_graph_input(df)
    nodes, inv = np.unique(np.concatenate([df["nameOrig"].astype(str), df["nameDest"].astype(str)]), return_inverse=True)
    src, dst = inv[: len(df)], inv[len(df):]
    edges = pd.DataFrame({"s": src, "d": dst}).drop_duplicates()
    comp = _components(edges["s"].to_numpy(), edges["d"].to_numpy(), len(nodes))
    sizes = np.bincount(np.unique(comp, return_inverse=True)[1])
    out_deg = edges.groupby("s").size()
    in_deg = edges.groupby("d").size()
    sender_counts = df.groupby("nameOrig").size()
    return {**facts, "transactions": int(len(df)), "nodes": int(len(nodes)), "distinct_edges": int(len(edges)),
            "senders": int(sender_counts.size), "receivers": int(df["nameDest"].nunique()),
            "sender_repeat_fraction": float((sender_counts > 1).mean()),
            "merchant_receiver_fraction": float(df["nameDest"].astype(str).str.startswith("M").mean()),
            "max_out_degree": int(out_deg.max()), "max_in_degree": int(in_deg.max()),
            "weak_components": int(len(sizes)), "largest_component_nodes": int(sizes.max()),
            "caveat": "Structure only. Sparse graphs (few repeated accounts) limit what any graph model can learn."}


def dest_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal receiver features using ONLY strictly earlier steps:
    dest_prev_txn_count (earlier transactions to this receiver) and
    dest_prev_unique_senders (distinct senders that had paid this receiver before)."""
    validate_graph_input(df)
    d = pd.DataFrame({"nameOrig": df["nameOrig"].astype(str).to_numpy(), "nameDest": df["nameDest"].astype(str).to_numpy(),
                      "step": df["step"].to_numpy()}, index=df.index)
    per = d.groupby(["nameDest", "step"]).size().rename("n").reset_index().sort_values(["nameDest", "step"])
    per["prev_n"] = per.groupby("nameDest")["n"].cumsum() - per["n"]
    first = d.groupby(["nameDest", "nameOrig"])["step"].min().rename("first_step").reset_index()
    first_sorted = {k: np.sort(v["first_step"].to_numpy()) for k, v in first.groupby("nameDest")}
    out = d[["nameDest", "step"]].merge(per[["nameDest", "step", "prev_n"]], on=["nameDest", "step"], how="left")
    uniq = np.array([int(np.searchsorted(first_sorted[nd], s, side="left")) for nd, s in zip(out["nameDest"], out["step"])])
    res = pd.DataFrame({"dest_prev_txn_count": out["prev_n"].to_numpy(dtype=float),
                        "dest_prev_unique_senders": uniq.astype(float)}, index=df.index)
    return res


def evaluate_graph_baseline(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> dict:
    """Ranking quality of each graph feature as a raw score on the held-out TEST time range (no fitting,
    so nothing is tuned on test). Compare with the tabular model's test PR-AUC from the SAME split."""
    if TARGET not in df.columns:
        raise GraphInputError(f"'{TARGET}' column required for evaluation")
    df = df.reset_index(drop=True)
    feats = dest_graph_features(df)
    _, _, te, steps = time_split(df, train_frac, val_frac)
    y = te[TARGET].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return {"status": "insufficient_data", "reason": "Test range lacks both classes.", "split_steps": steps}
    res = {c: ranking_metrics(y, feats.loc[te.index, c].to_numpy()) for c in feats.columns}
    return {"status": "ok", "split_steps": steps, "test_rows": int(len(te)), "features": res,
            "combination_with_xgboost": "not validated; evaluate any blend on the validation range, then report once on test"}
