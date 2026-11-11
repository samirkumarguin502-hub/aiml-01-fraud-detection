"""Reference profile + PSI used by monitoring.

Pure numpy/pandas implementation.

The profile stores binned proportions, not raw transaction rows:
- Feature distributions: TRAIN features.
- Prediction-score distribution: VALIDATION scores.

day_index is excluded because it grows monotonically with time.
PSI thresholds (0.1 / 0.25) are common rules of thumb, not
statistical significance tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_VERSION = 1

EXCLUDED_FROM_DRIFT = ("day_index",)

SCORE_EDGES = [
    round(i / 10, 1)
    for i in range(1, 10)
]

PSI_WARN = 0.10
PSI_ALERT = 0.25
EPS = 1e-4


def _as_finite_array(values, name: str) -> np.ndarray:
    """Convert values to a finite one-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


def _validate_edges(edges: list[float]) -> np.ndarray:
    """Validate histogram bin edges."""
    try:
        edge_array = np.asarray(edges, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Bin edges must be numeric."
        ) from exc

    if edge_array.ndim != 1:
        raise ValueError(
            "Bin edges must be one-dimensional."
        )

    if not np.isfinite(edge_array).all():
        raise ValueError(
            "Bin edges contain non-finite values."
        )

    if len(edge_array) > 1 and not np.all(
        np.diff(edge_array) > 0
    ):
        raise ValueError(
            "Bin edges must be strictly increasing."
        )

    return edge_array


def _bin_props(
    values: np.ndarray,
    edges: list[float],
) -> list[float]:
    """Calculate normalized proportions for histogram bins."""
    array = _as_finite_array(
        values,
        "values",
    )

    edge_array = _validate_edges(edges)

    indices = np.searchsorted(
        edge_array,
        array,
        side="right",
    )

    counts = np.bincount(
        indices,
        minlength=len(edge_array) + 1,
    ).astype(float)

    total = counts.sum()

    if total <= 0:
        raise ValueError(
            "Cannot calculate bin proportions for empty data."
        )

    return (counts / total).tolist()


def _edges_for(values: np.ndarray) -> list[float]:
    """Generate stable quantile-based feature bin edges."""
    array = _as_finite_array(
        values,
        "feature values",
    )

    unique_values = np.unique(array)

    if len(unique_values) <= 1:
        return []

    # Binary feature: one boundary separates 0 and 1.
    if np.isin(unique_values, [0.0, 1.0]).all():
        return [0.5]

    quantiles = np.quantile(
        array,
        np.linspace(0, 1, 11)[1:-1],
    )

    # Duplicate quantile boundaries are removed.
    edges = np.unique(quantiles)

    return edges.astype(float).tolist()


def build_reference_profile(
    X_train: pd.DataFrame,
    val_scores,
) -> dict:
    """Build a monitoring baseline from training features and validation scores.

    Feature distributions use training rows.
    Score distribution uses validation predictions.

    These two inputs intentionally do not need equal row counts.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if X_train.empty:
        raise ValueError(
            "X_train cannot be empty."
        )

    if not X_train.columns.is_unique:
        raise ValueError(
            "X_train contains duplicate feature names."
        )

    scores = _as_finite_array(
        val_scores,
        "validation scores",
    )

    # Model scores used for monitoring must be probabilities.
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(
            "Validation scores must be between 0 and 1."
        )

    features = {}

    for column in X_train.columns:
        if column in EXCLUDED_FROM_DRIFT:
            continue

        values = _as_finite_array(
            X_train[column].to_numpy(),
            f"feature '{column}'",
        )

        edges = _edges_for(values)

        features[str(column)] = {
            "edges": edges,
            "props": _bin_props(
                values,
                edges,
            ),
        }

    return {
        "version": PROFILE_VERSION,
        "n_train_rows": int(len(X_train)),
        "n_validation_scores": int(len(scores)),
        "excluded": list(EXCLUDED_FROM_DRIFT),
        "features": features,
        "score": {
            "edges": SCORE_EDGES.copy(),
            "props": _bin_props(
                scores,
                SCORE_EDGES,
            ),
        },
    }


def _validate_proportions(values, name: str) -> np.ndarray:
    """Validate and normalize histogram proportions."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric proportions."
        ) from exc

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a non-empty one-dimensional array."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if np.any(array < 0):
        raise ValueError(
            f"{name} cannot contain negative proportions."
        )

    total = float(array.sum())

    if total <= 0:
        raise ValueError(
            f"{name} must have a positive total."
        )

    return array / total


def psi(
    expected: list[float],
    actual: list[float],
) -> float:
    """Calculate Population Stability Index.

    PSI = sum((actual - expected) * ln(actual / expected))
    """
    expected_array = _validate_proportions(
        expected,
        "expected",
    )

    actual_array = _validate_proportions(
        actual,
        "actual",
    )

    if expected_array.shape != actual_array.shape:
        raise ValueError(
            "Expected and actual distributions must have "
            "the same number of bins."
        )

    expected_array = np.clip(
        expected_array,
        EPS,
        None,
    )

    actual_array = np.clip(
        actual_array,
        EPS,
        None,
    )

    return float(
        np.sum(
            (actual_array - expected_array)
            * np.log(actual_array / expected_array)
        )
    )


def psi_level(value: float) -> str:
    """Convert PSI into a monitoring status."""
    value = float(value)

    if not np.isfinite(value) or value < 0:
        raise ValueError(
            "PSI value must be finite and non-negative."
        )

    if value > PSI_ALERT:
        return "alert"

    if value > PSI_WARN:
        return "warning"

    return "stable"


def psi_against_profile(
    entry: dict,
    values,
) -> float:
    """Calculate PSI between a saved profile entry and current observations."""
    if not isinstance(entry, dict):
        raise ValueError(
            "Profile entry must be a dictionary."
        )

    if "edges" not in entry or "props" not in entry:
        raise ValueError(
            "Profile entry must contain 'edges' and 'props'."
        )

    edges = _validate_edges(
        entry["edges"]
    )

    actual_props = _bin_props(
        _as_finite_array(
            values,
            "current values",
        ),
        edges.tolist(),
    )

    return psi(
        entry["props"],
        actual_props,
    )