"""Metrics and threshold selection. Pure numpy/sklearn; no I/O.

Scores from class-weighted models are RANKING scores, not calibrated probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

DEFAULT_GRID = tuple(np.round(np.linspace(0.05, 0.95, 19), 2).tolist())


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def ranking_metrics(y_true, score) -> dict:
    """Threshold-free metrics. PR-AUC is average precision. None when undefined."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    has_pos, has_neg = bool(y.sum() > 0), bool((1 - y).sum() > 0)
    return {
        "pr_auc": float(average_precision_score(y, s)) if has_pos else None,
        "roc_auc": float(roc_auc_score(y, s)) if (has_pos and has_neg) else None,
        "prevalence": float(y.mean()) if len(y) else None,
        "n": int(len(y)), "n_fraud": int(y.sum()),
    }


def threshold_metrics(y_true, score, threshold: float) -> dict:
    """Confusion-matrix based metrics at one threshold (positive if score >= threshold)."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": _safe_div(2 * p * r, p + r),
            "flagged_rate": _safe_div(tp + fp, len(y))}


def compute_metrics(y_true, score, threshold: float) -> dict:
    return {**ranking_metrics(y_true, score), **threshold_metrics(y_true, score, threshold)}


def threshold_table(y_true, score, thresholds=DEFAULT_GRID) -> list[dict]:
    return [threshold_metrics(y_true, score, t) for t in thresholds]


def select_threshold(y_val, s_val, min_precision: float = 0.80) -> dict:
    """Pick the threshold on VALIDATION data only.

    Policy: among thresholds whose validation precision >= min_precision, take the one with the
    highest recall (ties -> higher threshold, i.e. fewer alerts). If none reaches min_precision,
    fall back to the max-F1 threshold and set fallback=True. Trade-off: a higher min_precision
    means fewer false alerts for analysts but more missed fraud.
    """
    y = np.asarray(y_val).astype(int)
    s = np.asarray(s_val, dtype=float)
    if y.sum() == 0:
        raise ValueError("Validation data has no fraud rows; cannot select a threshold.")
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    prec, rec, thr = precision_recall_curve(y, s)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    ok = prec >= min_precision
    fallback = not ok.any()
    if fallback:
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
        idx = int(np.argmax(f1))
    else:
        cand = np.flatnonzero(ok)
        best = cand[rec[cand] == rec[cand].max()]
        idx = int(best[np.argmax(thr[best])])
    m = threshold_metrics(y, s, float(thr[idx]))
    return {"threshold": float(thr[idx]), "policy": f"max_recall_at_precision>={min_precision}",
            "min_precision": float(min_precision), "fallback_to_max_f1": bool(fallback),
            "val_precision": m["precision"], "val_recall": m["recall"], "val_f1": m["f1"]}


def classification_report_dict(y_true, score, threshold: float) -> dict:
    """sklearn classification report at `threshold`, per class, with support. Classes absent from y_true
    get zeros (zero_division=0) rather than an error."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    rep = classification_report(y, pred, labels=[0, 1], target_names=["not_fraud", "fraud"],
                                output_dict=True, zero_division=0)
    keep = ("not_fraud", "fraud", "macro avg", "weighted avg")
    return {"threshold": float(threshold),
            **{k: {m: float(v) for m, v in rep[k].items()} for k in keep}}


def _downsample(*arrays, max_points: int):
    n = len(arrays[0])
    if n <= max_points:
        return [list(map(float, a)) for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [[float(a[i]) for i in idx] for a in arrays]


def curve_points(y_true, score, max_points: int = 200) -> dict | None:
    """Precision-recall and ROC curve points (evenly downsampled) for plotting. None if a class is missing."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    p, r = _downsample(prec, rec, max_points=max_points)
    f, t = _downsample(fpr, tpr, max_points=max_points)
    return {"pr": {"precision": p, "recall": r}, "roc": {"fpr": f, "tpr": t},
            "prevalence": float(y.mean()), "points_max": max_points}
