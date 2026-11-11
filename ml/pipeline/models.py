"""Model wrappers with one interface: fit / predict_score / save / load.

Kinds:
  xgboost   Primary gradient-boosting model; optional dependency, imported lazily.
  hist_gbm  Scikit-learn HistGradientBoosting CPU alternative.
  logreg    Standardised Logistic Regression baseline.

Class imbalance is computed from TRAIN labels only:
  xgboost  -> scale_pos_weight = n_negative / n_positive
  hist_gbm -> class_weight = "balanced"
  logreg   -> class_weight = "balanced"

Scores are ranking scores, not guaranteed calibrated probabilities.
All models run on CPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_KINDS = ("xgboost", "hist_gbm", "logreg")

XGB_DEFAULTS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
}

HGB_DEFAULTS = {
    "max_iter": 300,
    "learning_rate": 0.1,
}


class MissingDependencyError(RuntimeError):
    """Raised when an optional model dependency is unavailable."""


def library_versions() -> dict:
    """Return versions of the libraries used by the model pipeline."""
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }

    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = None

    return versions


def _import_xgboost():
    try:
        import xgboost
        return xgboost
    except ImportError as exc:
        raise MissingDependencyError(
            "xgboost is not installed in this environment. "
            "Install it into the active virtual environment using "
            "`python -m pip install xgboost`, or train with "
            "`--model hist_gbm` / `--model logreg`."
        ) from exc


def _validate_features(X: Any, name: str) -> None:
    """Validate a numeric feature matrix."""
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")

    if X.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")

    if X.shape[1] == 0:
        raise ValueError(f"{name} has no features.")

    if isinstance(X, pd.DataFrame):
        if X.columns.duplicated().any():
            raise ValueError(f"{name} contains duplicate feature names.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError(f"{name} must contain numeric columns.")

        values = X.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def _validate_labels(y: Any, expected_rows: int, name: str) -> np.ndarray:
    """Validate binary classification labels."""
    labels = np.asarray(y)

    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if len(labels) != expected_rows:
        raise ValueError(
            f"{name} has {len(labels)} rows but expected {expected_rows}."
        )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")

    labels = labels.astype(np.int32)

    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name} must contain both classes (0 and 1).")

    return labels


def _validate_feature_compatibility(X, X_val=None) -> None:
    if X_val is None:
        return

    if X.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )

    if isinstance(X, pd.DataFrame) and isinstance(X_val, pd.DataFrame):
        if list(X.columns) != list(X_val.columns):
            raise ValueError(
                "Training and validation feature names or ordering differ."
            )


class ScoreModel:
    """Unified interface for binary classification models."""

    def __init__(
        self,
        kind: str,
        params: dict | None = None,
        seed: int = 42,
    ):
        if kind not in MODEL_KINDS:
            raise ValueError(
                f"Unknown model kind {kind!r}; choose from {MODEL_KINDS}"
            )

        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary or None.")

        self.kind = kind
        self.seed = int(seed)
        self.params = dict(params or {})
        self.est = None
        self.fit_info: dict = {}
        self.params_used: dict = {}

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
    ) -> "ScoreModel":
        """Fit the model using training data and optional validation data."""

        _validate_features(X, "X")

        y = _validate_labels(y, X.shape[0], "y")

        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val and y_val must either both be provided or both be omitted."
            )

        if X_val is not None:
            _validate_features(X_val, "X_val")
            _validate_feature_compatibility(X, X_val)

            y_val = _validate_labels(
                y_val,
                X_val.shape[0],
                "y_val",
            )

        n_positive = int(np.sum(y == 1))
        n_negative = int(np.sum(y == 0))

        self.fit_info = {
            "training_rows": int(len(y)),
            "training_positive_rows": n_positive,
            "training_negative_rows": n_negative,
            "training_fraud_rate": float(n_positive / len(y)),
            "validation_rows": int(len(y_val)) if y_val is not None else 0,
            "class_imbalance_ratio": float(
                n_negative / max(n_positive, 1)
            ),
            "seed": self.seed,
        }

        if self.kind == "xgboost":
            self._fit_xgboost(
                X,
                y,
                X_val,
                y_val,
                n_positive,
                n_negative,
            )

        elif self.kind == "hist_gbm":
            self._fit_hist_gbm(X, y)

        elif self.kind == "logreg":
            self._fit_logreg(X, y)

        self.fit_info["model_kind"] = self.kind
        self.fit_info["library_versions"] = library_versions()

        return self

    def _fit_xgboost(
        self,
        X,
        y,
        X_val,
        y_val,
        n_positive: int,
        n_negative: int,
    ) -> None:
        xgb = _import_xgboost()

        params = {
            **XGB_DEFAULTS,
            "n_jobs": max(1, os.cpu_count() or 1),
            "importance_type": "gain",
            **self.params,
        }

        # Class imbalance must be calculated from training labels only.
        params["scale_pos_weight"] = (
            n_negative / max(n_positive, 1)
        )

        has_validation = X_val is not None and y_val is not None

        if not has_validation:
            params.pop("early_stopping_rounds", None)

        self.est = xgb.XGBClassifier(
            random_state=self.seed,
            **params,
        )

        if has_validation:
            self.est.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            best_iteration = getattr(
                self.est,
                "best_iteration",
                None,
            )

            self.fit_info["best_iteration"] = (
                int(best_iteration)
                if best_iteration is not None
                else None
            )

            self.fit_info["early_stopping_used"] = (
                params.get("early_stopping_rounds") is not None
            )
        else:
            self.est.fit(X, y)
            self.fit_info["best_iteration"] = None
            self.fit_info["early_stopping_used"] = False

        self.params_used = dict(params)
        self.fit_info["scale_pos_weight"] = float(
            params["scale_pos_weight"]
        )

    def _fit_hist_gbm(self, X, y) -> None:
        params = {
            **HGB_DEFAULTS,
            **self.params,
        }

        # Prevent caller-supplied values from overriding the fixed
        # imbalance strategy or introducing duplicate keyword arguments.
        params.pop("class_weight", None)
        params.pop("random_state", None)
        params.pop("early_stopping", None)

        self.est = HistGradientBoostingClassifier(
            random_state=self.seed,
            early_stopping=False,
            class_weight="balanced",
            **params,
        )

        self.est.fit(X, y)

        self.params_used = {
            **params,
            "class_weight": "balanced",
            "early_stopping": False,
        }

        self.fit_info["class_weight"] = "balanced"

    def _fit_logreg(self, X, y) -> None:
        params = {
            "max_iter": 1000,
            **self.params,
        }

        params.pop("class_weight", None)
        params.pop("random_state", None)

        classifier = LogisticRegression(
            class_weight="balanced",
            random_state=self.seed,
            **params,
        )

        self.est = make_pipeline(
            StandardScaler(),
            classifier,
        )

        self.est.fit(X, y)

        self.params_used = {
            **params,
            "class_weight": "balanced",
        }

        self.fit_info["class_weight"] = "balanced"

    def feature_importance(
        self,
        names: list[str],
        top: int = 20,
    ) -> dict | None:
        """Return top feature importance values when available.

        XGBoost: gain importance.
        Logistic Regression: absolute standardized coefficients.
        HistGradientBoosting: not provided by this wrapper.
        """

        if self.est is None:
            return None

        if top < 1:
            raise ValueError("top must be at least 1.")

        if self.kind == "xgboost":
            values = np.asarray(
                self.est.feature_importances_,
                dtype=float,
            )
            importance_type = "gain"

        elif self.kind == "logreg":
            values = np.abs(
                np.asarray(
                    self.est[-1].coef_[0],
                    dtype=float,
                )
            )
            importance_type = "abs_standardised_coefficient"

        else:
            return None

        if len(values) != len(names):
            return None

        if not np.isfinite(values).all():
            return None

        order = np.argsort(values)[::-1][:top]

        return {
            "type": importance_type,
            "features": [str(names[index]) for index in order],
            "values": [
                float(values[index])
                for index in order
            ],
        }

    def predict_score(self, X) -> np.ndarray:
        """Return positive-class ranking scores."""

        if self.est is None:
            raise RuntimeError("Model is not fitted.")

        _validate_features(X, "X")

        probabilities = self.est.predict_proba(X)

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise RuntimeError(
                "Model did not return binary classification scores."
            )

        scores = np.asarray(
            probabilities[:, 1],
            dtype=float,
        )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Model generated non-finite prediction scores."
            )

        return scores

    def save(self, directory: Path) -> str:
        """Save a fitted model to its native or joblib format."""

        if self.est is None:
            raise RuntimeError("Cannot save an unfitted model.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.kind == "xgboost":
            filename = "model.json"
            self.est.save_model(str(directory / filename))

        else:
            filename = "model.joblib"

            joblib.dump(
                {
                    "model_kind": self.kind,
                    "seed": self.seed,
                    "params": self.params,
                    "params_used": self.params_used,
                    "fit_info": self.fit_info,
                    "estimator": self.est,
                },
                directory / filename,
            )

        return filename

    @classmethod
    def load(
        cls,
        kind: str,
        path: Path,
        seed: int = 42,
    ) -> "ScoreModel":
        """Load a previously saved model."""

        if kind not in MODEL_KINDS:
            raise ValueError(
                f"Unknown model kind {kind!r}; choose from {MODEL_KINDS}"
            )

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact does not exist: {path}"
            )

        model = cls(kind, seed=seed)

        if kind == "xgboost":
            xgb = _import_xgboost()

            model.est = xgb.XGBClassifier()
            model.est.load_model(str(path))

        else:
            artifact = joblib.load(path)

            # Backward compatibility with earlier artifacts that stored
            # the estimator directly using joblib.dump(self.est, path).
            if (
                isinstance(artifact, dict)
                and artifact.get("model_kind") == kind
                and "estimator" in artifact
            ):
                model.est = artifact["estimator"]
                model.seed = int(artifact.get("seed", seed))
                model.params = dict(artifact.get("params", {}))
                model.params_used = dict(
                    artifact.get("params_used", {})
                )
                model.fit_info = dict(
                    artifact.get("fit_info", {})
                )
            else:
                model.est = artifact

        if model.est is None:
            raise RuntimeError(
                f"Failed to load model artifact: {path}"
            )

        return model