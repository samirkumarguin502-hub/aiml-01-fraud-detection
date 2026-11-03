"""Loads the model artifact once and reports exactly why it is unavailable if it is."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.errors import AppError
from backend.app.logging_setup import log_event
from ml.pipeline.model_io import ArtifactError, LoadedModel, load_artifact

log = logging.getLogger("fraudapp")


class ModelService:
    """state: 'unavailable' (nothing configured/found), 'invalid' (found but rejected), 'ready'."""

    def __init__(self, model_dir: Path | None):
        self.model_dir = model_dir
        self.state, self.reason = "unavailable", "MODEL_ARTIFACT_DIR is not set."
        self.loaded: LoadedModel | None = None

    def load(self) -> None:
        self.loaded = None
        d = self.model_dir
        if d is None:
            self.state, self.reason = "unavailable", "MODEL_ARTIFACT_DIR is not set."
            return
        d = Path(d)
        target = d if (d / "metadata.json").is_file() else (d / "primary" if (d / "primary" / "metadata.json").is_file() else None)
        if target is None:
            self.state = "unavailable"
            self.reason = ("No trained model artifact found at the configured location. Train one with "
                           "`python -m ml.train --data <csv>` and point MODEL_ARTIFACT_DIR at its 'primary' folder.")
            return
        try:
            self.loaded = load_artifact(target)
            self.state, self.reason = "ready", None
        except ArtifactError as e:
            self.state, self.reason = "invalid", str(e)
        except Exception as e:  # noqa: BLE001 - never leak internals; log type only
            self.state, self.reason = "invalid", "Model file could not be loaded (see server log)."
            log_event(log, "model_load_failed", logging.ERROR, exc_type=type(e).__name__)
        log_event(log, "model_load", model_state=self.state)

    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.loaded is not None

    def require_ready(self) -> LoadedModel:
        if not self.ready:
            raise AppError(503, "model_unavailable",
                           "No usable trained model is loaded, so predictions are unavailable.",
                           details={"model_state": self.state, "reason": self.reason})
        return self.loaded

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        m = self.require_ready()
        try:
            return m.predict_score(frame)
        except AppError:
            raise
        except Exception as e:  # noqa: BLE001
            log_event(log, "predict_failed", logging.ERROR, exc_type=type(e).__name__)
            raise AppError(500, "prediction_failed", "Prediction failed.") from None

    def info(self) -> dict:
        base = {"state": self.state, "reason": self.reason}
        if not self.ready:
            return {**base, "model": None}
        meta = self.loaded.metadata
        prov = meta.get("provenance", {})
        label = prov.get("data_label") or "unknown"
        return {**base, "model": {
            "kind": meta["model_kind"], "sha256": meta["model_sha256"], "created_utc": meta.get("created_utc"),
            "threshold": meta["threshold"], "feature_version": meta["feature_contract"]["version"],
            "feature_set": self.loaded.contract.feature_set, "input_fields": self.loaded.contract.input_columns,
            "model_version": meta.get("model_version"), "training": meta.get("training"),
            "feature_count": len(meta["feature_contract"]["feature_names"]),
            "type_vocabulary": self.loaded.contract.type_vocab,
            "library_versions": meta["library_versions"], "score_semantics": meta.get("score_semantics"),
            "metrics": {"origin": label, "metrics_are_synthetic": label.upper().startswith("SYNTH"),
                        "dataset_sha256": prov.get("dataset_sha256"), "split_steps": prov.get("split_steps"),
                        **meta.get("metrics", {})}}}
