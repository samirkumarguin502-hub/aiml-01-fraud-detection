"""Metrics and threshold selection. Pure numpy/sklearn; no I/O.

Scores from class-weighted models are ranking scores, not necessarily
calibrated probabilities.

Threshold selection must use VALIDATION data only. The test set must not
influence threshold selection or model tuning.
"""
from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_GRID = tuple(
    np.round(np.linspace(0.05, 0.95, 19), 2).tolist()
)


def _validate_inputs(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and finite prediction scores."""

    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and score must be one-dimensional arrays.")

    if len(y) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(y) != len(s):
        raise ValueError(
            f"Length mismatch: y_true has {len(y)} rows, "
            f"score has {len(s)} rows."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN or infinite values.")

    return y.astype(np.int32), s


def _validate_threshold(threshold: float) -> float:
    """Validate a finite threshold in the supported score range."""

    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def ranking_metrics(y_true, score) -> dict:
    """Threshold-free metrics. Undefined metrics are returned as None."""

    y, s = _validate_inputs(y_true, score)

    has_positive = bool(np.any(y == 1))
    has_negative = bool(np.any(y == 0))

    pr_auc = (
        float(average_precision_score(y, s))
        if has_positive
        else None
    )

    roc_auc = (
        float(roc_auc_score(y, s))
        if has_positive and has_negative
        else None
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "prevalence": float(y.mean()),
        "n": int(len(y)),
        "n_fraud": int(y.sum()),
    }


def threshold_metrics(
    y_true,
    score,
    threshold: float,
) -> dict:
    """Confusion-matrix metrics at one threshold.

    Positive prediction if score >= threshold.
    """

    y, s = _validate_inputs(y_true, score)
    threshold = _validate_threshold(threshold)

    predictions = (s >= threshold).astype(np.int32)

    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    f1 = _safe_div(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_rate": _safe_div(tp + fp, len(y)),
    }


def compute_metrics(
    y_true,
    score,
    threshold: float,
) -> dict:
    """Combine ranking and threshold-dependent metrics."""

    return {
        **ranking_metrics(y_true, score),
        **threshold_metrics(y_true, score, threshold),
    }


def threshold_table(
    y_true,
    score,
    thresholds=DEFAULT_GRID,
) -> list[dict]:
    """Evaluate a collection of thresholds."""

    y, s = _validate_inputs(y_true, score)

    thresholds = list(thresholds)

    if not thresholds:
        raise ValueError("At least one threshold is required.")

    validated = [
        _validate_threshold(value)
        for value in thresholds
    ]

    return [
        threshold_metrics(y, s, threshold)
        for threshold in validated
    ]


def select_threshold(
    y_val,
    s_val,
    min_precision: float = 0.80,
) -> dict:
    """Select the operating threshold using validation data only.

    Policy:
      1. Prefer the highest recall among thresholds satisfying minimum
         validation precision.
      2. Break recall ties by choosing the higher threshold.
      3. If no threshold satisfies minimum precision, use maximum F1.

    The selected threshold must be frozen before test evaluation.
    """

    y, scores = _validate_inputs(y_val, s_val)

    min_precision = float(min_precision)

    if not np.isfinite(min_precision):
        raise ValueError("min_precision must be finite.")

    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")

    if not np.any(y == 1):
        raise ValueError(
            "Validation data has no fraud rows; "
            "cannot select a threshold."
        )

    if not np.any(y == 0):
        raise ValueError(
            "Validation data has no legitimate rows; "
            "precision-constrained threshold selection is unreliable."
        )

    precision, recall, thresholds = precision_recall_curve(
        y,
        scores,
    )

    # The final precision-recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        raise ValueError(
            "No candidate thresholds available for validation."
        )

    valid = (
        np.isfinite(precision)
        & np.isfinite(recall)
        & np.isfinite(thresholds)
    )

    precision = precision[valid]
    recall = recall[valid]
    thresholds = thresholds[valid]

    if len(thresholds) == 0:
        raise ValueError(
            "No valid candidate thresholds available."
        )

    eligible = precision >= min_precision
    fallback = not bool(np.any(eligible))

    if fallback:
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )

        # np.argmax returns the first maximum.
        # Explicit tie handling selects the higher threshold.
        best_f1 = f1.max()
        candidates = np.flatnonzero(f1 == best_f1)

        best_index = int(
            candidates[np.argmax(thresholds[candidates])]
        )

        policy = "max_f1_fallback"

    else:
        candidates = np.flatnonzero(eligible)

        best_recall = recall[candidates].max()

        best_candidates = candidates[
            recall[candidates] == best_recall
        ]

        best_index = int(
            best_candidates[
                np.argmax(thresholds[best_candidates])
            ]
        )

        policy = f"max_recall_at_precision>={min_precision}"

    selected_threshold = float(thresholds[best_index])

    metrics = threshold_metrics(
        y,
        scores,
        selected_threshold,
    )

    return {
        "threshold": selected_threshold,
        "policy": policy,
        "min_precision": min_precision,
        "fallback_to_max_f1": bool(fallback),
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "validation_rows": int(len(y)),
        "validation_fraud_rows": int(y.sum()),
    }


def classification_report_dict(
    y_true,
    score,
    threshold: float,
) -> dict:
    """Return a classification report with fixed binary class labels."""

    y, s = _validate_inputs(y_true, score)
    threshold = _validate_threshold(threshold)

    predictions = (s >= threshold).astype(np.int32)

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["not_fraud", "fraud"],
        output_dict=True,
        zero_division=0,
    )

    keep = (
        "not_fraud",
        "fraud",
        "macro avg",
        "weighted avg",
    )

    return {
        "threshold": threshold,
        **{
            key: {
                metric: float(value)
                for metric, value in report[key].items()
            }
            for key in keep
        },
    }


def _downsample(
    *arrays,
    max_points: int,
) -> list[list[float]]:
    """Downsample equally spaced indices while preserving endpoints."""

    if not arrays:
        return []

    lengths = {len(array) for array in arrays}

    if len(lengths) != 1:
        raise ValueError(
            "All arrays passed to _downsample must have equal lengths."
        )

    n = len(arrays[0])

    if n == 0:
        return [[] for _ in arrays]

    if max_points < 2:
        raise ValueError("max_points must be at least 2.")

    if n <= max_points:
        return [
            list(map(float, array))
            for array in arrays
        ]

    indices = np.unique(
        np.linspace(
            0,
            n - 1,
            max_points,
        ).round().astype(int)
    )

    return [
        [float(array[index]) for index in indices]
        for array in arrays
    ]


def curve_points(
    y_true,
    score,
    max_points: int = 200,
) -> dict | None:
    """Generate downsampled PR and ROC curve coordinates.

    Returns None when either class is missing because ROC analysis
    requires both positive and negative observations.
    """

    y, s = _validate_inputs(y_true, score)

    if not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer.")

    if max_points < 2:
        raise ValueError("max_points must be at least 2.")

    if len(np.unique(y)) < 2:
        return None

    precision, recall, _ = precision_recall_curve(
        y,
        s,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y,
        s,
    )

    pr_precision, pr_recall = _downsample(
        precision,
        recall,
        max_points=max_points,
    )

    roc_fpr, roc_tpr = _downsample(
        false_positive_rate,
        true_positive_rate,
        max_points=max_points,
    )

    return {
        "pr": {
            "precision": pr_precision,
            "recall": pr_recall,
        },
        "roc": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
        },
        "prevalence": float(y.mean()),
        "points_max": int(max_points),
    }