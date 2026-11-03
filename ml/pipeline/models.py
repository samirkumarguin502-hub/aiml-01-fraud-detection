"""Model wrappers with one interface: fit / predict_score / save / load.

Kinds:
  xgboost   primary model (optional dependency; imported lazily, never substituted silently)
  hist_gbm  scikit-learn HistGradientBoosting (CPU gradient boosting; same family, no extra deps)
  logreg    standardised logistic regression baseline

Class imbalance (computed from TRAIN labels only): xgboost uses scale_pos_weight = n_neg/n_pos;
hist_gbm and logreg use class_weight="balanced". Resulting scores are not calibrated probabilities.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KINDS = ("xgboost", "hist_gbm", "logreg")
XGB_DEFAULTS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "aucpr",
                "early_stopping_rounds": 30}
HGB_DEFAULTS = {"max_iter": 300, "learning_rate": 0.1}


class MissingDependencyError(RuntimeError):
    pass


def library_versions() -> dict:
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
         "sklearn": sklearn.__version__, "joblib": joblib.__version__}
    try:
        import xgboost
        v["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return v


def _import_xgboost():
    try:
        import xgboost
        return xgboost
    except ImportError as e:
        raise MissingDependencyError(
            "xgboost is not installed in this environment. Install it into your existing venv "
            "(e.g. `python -m pip install xgboost`) or train with --model hist_gbm / logreg.") from e


class ScoreModel:
    def __init__(self, kind: str, params: dict | None = None, seed: int = 42):
        if kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind {kind!r}; choose from {MODEL_KINDS}")
        self.kind, self.seed = kind, seed
        self.params = dict(params or {})
        self.est = None
        self.fit_info: dict = {}

    def fit(self, X, y, X_val=None, y_val=None) -> "ScoreModel":
        y = np.asarray(y).astype(int)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            raise ValueError("Training labels must contain both classes.")
        if self.kind == "xgboost":
            xgb = _import_xgboost()
            p = {**XGB_DEFAULTS, "n_jobs": max(1, os.cpu_count() or 1), **self.params}
            have_val = X_val is not None and y_val is not None
            if not have_val:
                p.pop("early_stopping_rounds", None)
            p["scale_pos_weight"] = (len(y) - n_pos) / n_pos
            self.est = xgb.XGBClassifier(random_state=self.seed, **p)
            if have_val:
                self.est.fit(X, y, eval_set=[(X_val, np.asarray(y_val).astype(int))], verbose=False)
                self.fit_info["best_iteration"] = getattr(self.est, "best_iteration", None)
            else:
                self.est.fit(X, y)
            self.fit_info["scale_pos_weight"] = float(p["scale_pos_weight"])
            self.params_used = p
        elif self.kind == "hist_gbm":
            p = {**HGB_DEFAULTS, **self.params}
            self.est = HistGradientBoostingClassifier(random_state=self.seed, early_stopping=False,
                                                      class_weight="balanced", **p)
            self.est.fit(X, y)
            self.params_used = {**p, "class_weight": "balanced"}
        else:
            p = {"max_iter": 1000, **self.params}
            self.est = make_pipeline(StandardScaler(), LogisticRegression(
                class_weight="balanced", random_state=self.seed, **p))
            self.est.fit(X, y)
            self.params_used = {**p, "class_weight": "balanced"}
        return self

    def feature_importance(self, names: list[str], top: int = 20) -> dict | None:
        """Top-N importances. XGBoost: gain; logistic regression: |standardised coefficient|; the
        scikit-learn GBM exposes none cheaply, so None (no permutation importance is computed)."""
        if self.est is None:
            return None
        if self.kind == "xgboost":
            vals, kind = np.asarray(self.est.feature_importances_, dtype=float), "gain"
        elif self.kind == "logreg":
            vals, kind = np.abs(np.asarray(self.est[-1].coef_[0], dtype=float)), "abs_standardised_coefficient"
        else:
            return None
        if len(vals) != len(names):
            return None
        order = np.argsort(vals)[::-1][:top]
        return {"type": kind, "features": [names[i] for i in order], "values": [float(vals[i]) for i in order]}

    def predict_score(self, X) -> np.ndarray:
        if self.est is None:
            raise RuntimeError("Model is not fitted.")
        return np.asarray(self.est.predict_proba(X)[:, 1], dtype=float)

    def save(self, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        if self.kind == "xgboost":
            name = "model.json"  # native XGBoost format, not pickle
            self.est.save_model(str(directory / name))
        else:
            name = "model.joblib"  # pickle-based: only load artifacts you created (sha256-checked)
            joblib.dump(self.est, directory / name)
        return name

    @classmethod
    def load(cls, kind: str, path: Path, seed: int = 42) -> "ScoreModel":
        m = cls(kind, seed=seed)
        if kind == "xgboost":
            xgb = _import_xgboost()
            m.est = xgb.XGBClassifier()
            m.est.load_model(str(path))
        else:
            m.est = joblib.load(path)
        return m
