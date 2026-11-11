"""Framework-free monitoring computations over real stored rows.

Never invents monitoring numbers. When data is insufficient, the report
states that explicitly.

Drift uses PSI against the reference profile stored in the model artifact.
Performance metrics use analyst-labelled transactions only.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.pipeline.evaluate import threshold_metrics
from ml.pipeline.features import build_features
from ml.pipeline.reference import (
    PSI_ALERT,
    PSI_WARN,
    psi_against_profile,
    psi_level,
)


MIN_DRIFT_ROWS = 100
MIN_LABELS = 20

RETRAIN_NOTE = (
    "A drift alert is a signal for a human to look, not an action. "
    "It does not retrain or replace the active model; retraining is a "
    "separate, explicit, audited workflow."
)

SCORED_REQUIRED_COLUMNS = (
    "score",
)

LABELLED_REQUIRED_COLUMNS = (
    "score",
    "threshold",
    "label",
)


def _validate_minimum(value: int, name: str) -> int:
    """Validate a minimum observation count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_numeric(
    series: pd.Series,
    name: str,
) -> np.ndarray:
    """Convert a Series to finite numeric values."""
    try:
        values = pd.to_numeric(
            series,
            errors="raise",
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Column '{name}' must contain numeric values."
        ) from exc

    if not np.isfinite(values).all():
        raise ValueError(
            f"Column '{name}' contains missing or non-finite values."
        )

    return values


def _require_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def _validate_profile(profile: dict) -> None:
    """Validate the stored reference-profile structure."""
    if not isinstance(profile, dict):
        raise ValueError("Reference profile must be a dictionary.")

    if not isinstance(profile.get("features"), dict):
        raise ValueError(
            "Reference profile has no valid feature distributions."
        )

    if not isinstance(profile.get("score"), dict):
        raise ValueError(
            "Reference profile has no valid score distribution."
        )


def drift_report(
    rows: pd.DataFrame,
    loaded_model,
    *,
    min_rows: int = MIN_DRIFT_ROWS,
) -> dict:
    """Calculate feature and score drift over recent scored transactions.

    Expected rows:
        model input columns + score

    Transactions are processed in their stored insertion order.
    Rows missing required model inputs are excluded from comparison.
    """
    min_rows = _validate_minimum(
        min_rows,
        "min_rows",
    )

    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame.")

    if loaded_model is None:
        return {
            "method": "PSI",
            "status": "no_model",
            "rows_compared": 0,
            "min_rows": min_rows,
            "reason": "No usable model is loaded.",
        }

    contract = loaded_model.contract
    metadata = loaded_model.metadata

    profile = metadata.get("reference_profile")

    base = {
        "method": "PSI",
        "thresholds": {
            "warning": PSI_WARN,
            "alert": PSI_ALERT,
        },
        "rows_received": int(len(rows)),
        "rows_compared": 0,
        "rows_skipped": 0,
        "min_rows": min_rows,
        "threshold_note": (
            "Heuristic rule of thumb, not a significance test."
        ),
    }

    if profile is None:
        return {
            **base,
            "status": "no_reference",
            "reason": (
                "This artifact has no reference profile "
                "(trained before monitoring existed). "
                "Retrain to create one."
            ),
        }

    _validate_profile(profile)

    required_inputs = list(contract.input_columns)

    missing = [
        column for column in required_inputs
        if column not in rows.columns
    ]

    if missing:
        return {
            **base,
            "status": "missing_columns",
            "reason": (
                "Recent scored transactions do not contain all "
                f"model input columns: {missing}"
            ),
        }

    _require_columns(
        rows,
        SCORED_REQUIRED_COLUMNS,
        "rows",
    )

    # Keep only records with all required model inputs and a valid score.
    candidate = rows.copy()

    score_values = pd.to_numeric(
        candidate["score"],
        errors="coerce",
    )

    valid_mask = (
        candidate[required_inputs].notna().all(axis=1)
        & score_values.notna()
        & np.isfinite(
            score_values.fillna(0).to_numpy(dtype=float)
        )
    )

    candidate = candidate.loc[valid_mask].copy()
    candidate["score"] = score_values.loc[valid_mask].astype(float)

    base["rows_compared"] = int(len(candidate))
    base["rows_skipped"] = int(len(rows) - len(candidate))

    if len(candidate) < min_rows:
        return {
            **base,
            "status": "insufficient_data",
            "reason": (
                f"Need at least {min_rows} scored transactions; "
                f"have {len(candidate)}."
            ),
        }

    scores = candidate["score"].to_numpy(dtype=float)

    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        return {
            **base,
            "status": "invalid_scores",
            "reason": (
                "Stored prediction scores must be finite "
                "values between 0 and 1."
            ),
        }

    try:
        X = build_features(
            candidate.reset_index(drop=True),
            contract,
        )

        feature_results = []

        for name, entry in profile["features"].items():
            if name not in X.columns:
                continue

            values = X[name].to_numpy(dtype=float)

            value = psi_against_profile(
                entry,
                values,
            )

            feature_results.append({
                "feature": name,
                "psi": round(value, 4),
                "level": psi_level(value),
            })

        score_psi = psi_against_profile(
            profile["score"],
            scores,
        )

    except (ValueError, TypeError, KeyError) as exc:
        return {
            **base,
            "status": "calculation_error",
            "reason": (
                f"Could not calculate drift: {type(exc).__name__}: {exc}"
            ),
        }

    feature_results.sort(
        key=lambda item: item["psi"],
        reverse=True,
    )

    return {
        **base,
        "status": "ok",
        "features": feature_results,
        "excluded": profile.get("excluded", []),
        "score": {
            "psi": round(score_psi, 4),
            "level": psi_level(score_psi),
            "reference": "validation-set scores of the active model",
        },
        "reference_rows": profile.get("n_train_rows"),
        "reference_validation_scores": profile.get(
            "n_validation_scores"
        ),
    }


def quality_report(
    rows: pd.DataFrame,
    loaded_model,
    rejections: dict,
) -> dict:
    """Summarize stored transaction quality and request rejections."""
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame.")

    n = len(rows)

    unknown = 0

    if n and loaded_model is not None:
        if "type" not in rows.columns:
            raise ValueError(
                "rows must contain 'type' for quality monitoring."
            )

        unknown = int(
            (~rows["type"].isin(
                loaded_model.contract.type_vocab
            )).sum()
        )

    if not isinstance(rejections, dict):
        raise TypeError("rejections must be a dictionary.")

    rejection_counts = {}

    for code, count in rejections.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 0
        ):
            raise ValueError(
                f"Invalid rejection count for {code!r}."
            )

        rejection_counts[str(code)] = int(count)

    return {
        "rows": int(n),
        "unknown_type_rows": unknown,
        "unknown_type_rate": (
            unknown / n if n else None
        ),
        "rejected_requests_by_code": rejection_counts,
        "rejected_requests_total": int(
            sum(rejection_counts.values())
        ),
        "note": (
            "Rejected requests are counted by error code only; "
            "their content is never stored."
        ),
    }


def performance_report(
    labelled: pd.DataFrame,
    *,
    min_labels: int = MIN_LABELS,
) -> dict:
    """Calculate threshold metrics using analyst-labelled transactions.

    Expected columns:
        score, threshold, label

    The labelled data is not assumed to be a random sample.
    """
    min_labels = _validate_minimum(
        min_labels,
        "min_labels",
    )

    _require_columns(
        labelled,
        LABELLED_REQUIRED_COLUMNS,
        "labelled",
    )

    n = len(labelled)

    warning = (
        "Computed only on transactions an analyst labelled. "
        "Alerts are more likely to be reviewed than non-alerts, "
        "so this is a biased sample and not an estimate of "
        "overall model accuracy."
    )

    if n < min_labels:
        return {
            "status": "insufficient_labels",
            "labelled": int(n),
            "min_labels": min_labels,
            "reason": (
                f"Need at least {min_labels} labelled transactions "
                "containing both classes."
            ),
            "warning": warning,
        }

    labels = _finite_numeric(
        labelled["label"],
        "label",
    )

    scores = _finite_numeric(
        labelled["score"],
        "score",
    )

    thresholds = _finite_numeric(
        labelled["threshold"],
        "threshold",
    )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Analyst labels must contain only 0 and 1."
        )

    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError(
            "Analyst labels must contain only 0 and 1."
        )

    if len(np.unique(labels)) < 2:
        return {
            "status": "insufficient_labels",
            "labelled": int(n),
            "min_labels": min_labels,
            "reason": (
                f"Need at least {min_labels} labelled transactions "
                "containing both classes."
            ),
            "warning": warning,
        }

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(
            "Analyst-labelled prediction scores must be between 0 and 1."
        )

    if np.any(thresholds < 0) or np.any(thresholds > 1):
        raise ValueError(
            "Stored decision thresholds must be between 0 and 1."
        )

    # Existing behaviour: use the latest stored threshold.
    threshold = float(thresholds[-1])

    metrics = threshold_metrics(
        labels,
        scores,
        threshold,
    )

    return {
        "status": "ok",
        "labelled": int(n),
        "threshold": threshold,
        "warning": warning,
        **{
            key: metrics[key]
            for key in (
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "f1",
            )
        },
    }


def build_report(
    *,
    rows: pd.DataFrame,
    volume: dict,
    rejections: dict,
    labelled: pd.DataFrame,
    loaded_model,
    model_state: str,
    min_rows: int = MIN_DRIFT_ROWS,
) -> dict:
    """Build the complete monitoring report."""
    output = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "volume": volume,
        "retraining_note": RETRAIN_NOTE,
        "evidently": {
            "status": "not_requested",
        },
    }

    if loaded_model is None:
        return {
            **output,
            "status": "no_model",
            "model_state": model_state,
            "reason": (
                "No usable model is loaded, so drift and "
                "performance cannot be assessed."
            ),
            "data_quality": quality_report(
                rows,
                None,
                rejections,
            ),
            "drift": None,
            "performance": None,
            "drift_alert": False,
            "alert_reasons": [],
            "retraining_recommended": False,
        }

    drift = drift_report(
        rows,
        loaded_model,
        min_rows=min_rows,
    )

    performance = performance_report(
        labelled
    )

    reasons = []

    if drift["status"] == "ok":
        for feature in drift["features"]:
            if feature["level"] == "alert":
                reasons.append(
                    f"feature '{feature['feature']}' "
                    f"PSI {feature['psi']}"
                )

        if drift["score"]["level"] == "alert":
            reasons.append(
                f"score distribution PSI {drift['score']['psi']}"
            )

    status = (
        drift["status"]
        if drift["status"] != "ok"
        else "ok"
    )

    return {
        **output,
        "status": status,
        "model_state": model_state,
        "data_quality": quality_report(
            rows,
            loaded_model,
            rejections,
        ),
        "drift": drift,
        "performance": performance,
        "drift_alert": bool(reasons),
        "alert_reasons": reasons,
        "retraining_recommended": bool(reasons),
    }