"""Framework-free monitoring computations over REAL stored rows. Never invents numbers:
when there is not enough data the result says so explicitly."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.pipeline.evaluate import threshold_metrics
from ml.pipeline.features import build_features
from ml.pipeline.reference import PSI_ALERT, PSI_WARN, psi_against_profile, psi_level

MIN_DRIFT_ROWS = 100
MIN_LABELS = 20
RETRAIN_NOTE = ("A drift alert is a signal for a human to look, not an action. It does not retrain or "
                "replace the active model; retraining is a separate, explicit, audited workflow.")


def drift_report(rows: pd.DataFrame, loaded_model, *, min_rows: int = MIN_DRIFT_ROWS) -> dict:
    """rows: recent scored transactions (step,type,amount,nameOrig,nameDest,score) in insertion order."""
    meta = loaded_model.metadata
    prof = meta.get("reference_profile")
    base = {"method": "PSI", "thresholds": {"warning": PSI_WARN, "alert": PSI_ALERT},
            "rows_compared": int(len(rows)), "min_rows": min_rows,
            "threshold_note": "Heuristic rule of thumb, not a significance test."}
    need = [c for c in loaded_model.contract.input_columns if c in rows.columns]
    rows = rows.dropna(subset=need)  # rows scored by an older model without the needed balances are skipped
    base["rows_compared"] = int(len(rows))
    if prof is None:
        return {**base, "status": "no_reference",
                "reason": "This artifact has no reference profile (trained before monitoring existed). Retrain to create one."}
    if len(rows) < min_rows:
        return {**base, "status": "insufficient_data",
                "reason": f"Need at least {min_rows} scored transactions; have {len(rows)}."}
    X = build_features(rows.reset_index(drop=True), loaded_model.contract)
    feats = []
    for name, entry in prof["features"].items():
        if name in X.columns:
            v = psi_against_profile(entry, X[name].to_numpy(dtype=float))
            feats.append({"feature": name, "psi": round(v, 4), "level": psi_level(v)})
    sc = psi_against_profile(prof["score"], rows["score"].to_numpy(dtype=float))
    feats.sort(key=lambda f: f["psi"], reverse=True)
    return {**base, "status": "ok", "features": feats, "excluded": prof.get("excluded", []),
            "score": {"psi": round(sc, 4), "level": psi_level(sc),
                      "reference": "validation-set scores of the active model"},
            "reference_rows": prof.get("n_train_rows")}


def quality_report(rows: pd.DataFrame, loaded_model, rejections: dict) -> dict:
    n = len(rows)
    unknown = int((~rows["type"].isin(loaded_model.contract.type_vocab)).sum()) if (n and loaded_model) else 0
    return {"rows": n, "unknown_type_rows": unknown,
            "unknown_type_rate": (unknown / n) if n else None,
            "rejected_requests_by_code": rejections,
            "rejected_requests_total": int(sum(rejections.values())),
            "note": "Rejected requests are counted by error code only; their content is never stored."}


def performance_report(labelled: pd.DataFrame, *, min_labels: int = MIN_LABELS) -> dict:
    """labelled: columns score, threshold, label (analyst feedback). Not a random sample."""
    n = len(labelled)
    warn = ("Computed only on transactions an analyst labelled. Alerts are more likely to be reviewed than "
            "non-alerts, so this is a biased sample and not an estimate of overall model accuracy.")
    if n < min_labels or labelled["label"].nunique() < 2:
        return {"status": "insufficient_labels", "labelled": int(n), "min_labels": min_labels,
                "reason": f"Need at least {min_labels} labelled transactions containing both classes.", "warning": warn}
    thr = float(labelled["threshold"].iloc[-1])
    m = threshold_metrics(labelled["label"].to_numpy(), labelled["score"].to_numpy(), thr)
    return {"status": "ok", "labelled": int(n), "threshold": thr, "warning": warn,
            **{k: m[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1")}}


def build_report(*, rows: pd.DataFrame, volume: dict, rejections: dict, labelled: pd.DataFrame,
                 loaded_model, model_state: str, min_rows: int = MIN_DRIFT_ROWS) -> dict:
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "volume": volume, "retraining_note": RETRAIN_NOTE, "evidently": {"status": "not_requested"}}
    if loaded_model is None:
        return {**out, "status": "no_model", "model_state": model_state,
                "reason": "No usable model is loaded, so drift and performance cannot be assessed.",
                "data_quality": quality_report(rows, None, rejections), "drift": None, "performance": None,
                "drift_alert": False, "alert_reasons": []}
    drift = drift_report(rows, loaded_model, min_rows=min_rows)
    perf = performance_report(labelled)
    reasons = []
    if drift["status"] == "ok":
        reasons += [f"feature '{f['feature']}' PSI {f['psi']}" for f in drift["features"] if f["level"] == "alert"]
        if drift["score"]["level"] == "alert":
            reasons.append(f"score distribution PSI {drift['score']['psi']}")
    status = drift["status"] if drift["status"] != "ok" else "ok"
    return {**out, "status": status, "model_state": model_state,
            "data_quality": quality_report(rows, loaded_model, rejections), "drift": drift,
            "performance": perf, "drift_alert": bool(reasons), "alert_reasons": reasons,
            "retraining_recommended": bool(reasons)}
