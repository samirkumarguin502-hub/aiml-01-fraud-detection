"""Loads the model artifact once and reports why it is unavailable."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.errors import AppError
from backend.app.logging_setup import log_event
from ml.pipeline.model_io import ArtifactError, LoadedModel, load_artifact


log = logging.getLogger("fraudapp")


class ModelService:
    """Manage the active fraud detection model.

    States:
        unavailable: No model is configured or found.
        invalid: An artifact was found but rejected or failed to load.
        ready: A valid model artifact is loaded.
    """

    def __init__(self, model_dir: Path | None):
        self.model_dir = model_dir
        self.state = "unavailable"
        self.reason = "MODEL_ARTIFACT_DIR is not set."
        self.loaded: LoadedModel | None = None
        self._lock = threading.RLock()

    def load(self) -> None:
        """Load the configured artifact.

        Supports both:
            MODEL_ARTIFACT_DIR pointing directly to primary/
            MODEL_ARTIFACT_DIR pointing to a run directory containing primary/
        """

        with self._lock:
            self.loaded = None

            if self.model_dir is None:
                self.state = "unavailable"
                self.reason = "MODEL_ARTIFACT_DIR is not set."

                log_event(
                    log,
                    "model_load",
                    model_state=self.state,
                )
                return

            directory = Path(self.model_dir)

            try:
                if (directory / "metadata.json").is_file():
                    target = directory

                elif (
                    directory / "primary" / "metadata.json"
                ).is_file():
                    target = directory / "primary"

                else:
                    target = None

                if target is None:
                    self.state = "unavailable"
                    self.reason = (
                        "No trained model artifact found at the configured "
                        "location. Train one with "
                        "`python -m ml.train --data <csv>` and point "
                        "MODEL_ARTIFACT_DIR at its primary folder."
                    )

                    log_event(
                        log,
                        "model_load",
                        model_state=self.state,
                    )
                    return

                loaded = load_artifact(target)

                if loaded is None:
                    raise ArtifactError(
                        "Artifact loader returned no model."
                    )

                if not hasattr(loaded, "contract"):
                    raise ArtifactError(
                        "Loaded artifact has no feature contract."
                    )

                if not hasattr(loaded, "predict_score"):
                    raise ArtifactError(
                        "Loaded artifact has no prediction interface."
                    )

                if not hasattr(loaded, "threshold"):
                    raise ArtifactError(
                        "Loaded artifact has no decision threshold."
                    )

                self.loaded = loaded
                self.state = "ready"
                self.reason = None

                log_event(
                    log,
                    "model_load",
                    model_state=self.state,
                )

            except ArtifactError as exc:
                self.loaded = None
                self.state = "invalid"

                # Do not expose raw artifact errors or filesystem paths.
                self.reason = (
                    "The configured model artifact is corrupted, tampered, or incompatible; SHA-256 verification may have failed, or an artifact version mismatch may exist. "
                    "Check the server logs and artifact metadata."
                )

                log_event(
                    log,
                    "model_load_failed",
                    logging.ERROR,
                    exc_type=type(exc).__name__,
                )

                log_event(
                    log,
                    "model_load",
                    model_state=self.state,
                )

            except Exception as exc:
                self.loaded = None
                self.state = "invalid"
                self.reason = (
                    "Model file could not be loaded. "
                    "Check the server logs."
                )

                log_event(
                    log,
                    "model_load_failed",
                    logging.ERROR,
                    exc_type=type(exc).__name__,
                )

                log_event(
                    log,
                    "model_load",
                    model_state=self.state,
                )

    @property
    def ready(self) -> bool:
        """Return whether a usable model is currently loaded."""

        with self._lock:
            return (
                self.state == "ready"
                and self.loaded is not None
            )

    def require_ready(self) -> LoadedModel:
        """Return the loaded model or raise an application error."""

        with self._lock:
            if (
                self.state != "ready"
                or self.loaded is None
            ):
                raise AppError(
                    503,
                    "model_unavailable",
                    (
                        "No usable trained model is loaded, "
                        "so predictions are unavailable."
                    ),
                    details={
                        "model_state": self.state,
                        "reason": self.reason,
                    },
                )

            return self.loaded

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Generate and validate fraud ranking scores.

        Scores must:
            - Match the number of input transactions.
            - Be one-dimensional.
            - Contain only finite numeric values.
            - Be within the [0, 1] range expected by the platform.
        """

        model = self.require_ready()

        if not isinstance(frame, pd.DataFrame):
            raise AppError(
                422,
                "validation_error",
                "Prediction input must be a pandas DataFrame.",
            )

        if frame.empty:
            raise AppError(
                422,
                "validation_error",
                "Prediction input cannot be empty.",
            )

        try:
            raw_scores = model.predict_score(frame)

            scores = np.asarray(
                raw_scores,
                dtype=np.float64,
            )

            if scores.ndim != 1:
                raise ValueError(
                    "Model output must be one-dimensional."
                )

            if len(scores) != len(frame):
                raise ValueError(
                    "Model output length does not match input length."
                )

            if not np.isfinite(scores).all():
                raise ValueError(
                    "Model output contains non-finite values."
                )

            if (
                np.any(scores < 0.0)
                or np.any(scores > 1.0)
            ):
                raise ValueError(
                    "Model output contains values outside [0, 1]."
                )

            return scores

        except AppError:
            raise

        except Exception as exc:
            log_event(
                log,
                "predict_failed",
                logging.ERROR,
                exc_type=type(exc).__name__,
            )

            raise AppError(
                500,
                "prediction_failed",
                "Prediction failed.",
            ) from None

    def info(self) -> dict:
        """Return safe model metadata for health and dashboard endpoints."""

        with self._lock:
            state = self.state
            reason = self.reason
            loaded = self.loaded

        base = {
            "state": state,
            "reason": reason,
        }

        if state != "ready" or loaded is None:
            return {
                **base,
                "model": None,
            }

        metadata = loaded.metadata

        if not isinstance(metadata, dict):
            return {
                **base,
                "state": "invalid",
                "reason": "Loaded model metadata is invalid.",
                "model": None,
            }

        provenance = metadata.get("provenance") or {}

        if not isinstance(provenance, dict):
            provenance = {}

        data_label = provenance.get("data_label") or "unknown"

        if not isinstance(data_label, str):
            data_label = "unknown"

        feature_contract = metadata.get("feature_contract") or {}

        if not isinstance(feature_contract, dict):
            feature_contract = {}

        metrics = metadata.get("metrics") or {}

        if not isinstance(metrics, dict):
            metrics = {}

        library_versions = metadata.get("library_versions") or {}

        if not isinstance(library_versions, dict):
            library_versions = {}

        feature_names = feature_contract.get("feature_names") or []

        if not isinstance(feature_names, (list, tuple)):
            feature_names = []

        model_info = {
            "kind": metadata.get("model_kind"),
            "sha256": metadata.get("model_sha256"),
            "created_utc": metadata.get("created_utc"),
            "threshold": metadata.get("threshold"),
            "feature_version": feature_contract.get("version"),
            "feature_set": loaded.contract.feature_set,
            "input_fields": list(
                loaded.contract.input_columns
            ),
            "model_version": metadata.get("model_version"),
            "training": metadata.get("training"),
            "feature_count": len(feature_names),
            "type_vocabulary": list(
                loaded.contract.type_vocab
            ),
            "library_versions": library_versions,
            "score_semantics": metadata.get("score_semantics"),
            "metrics": {
                "origin": data_label,
                "metrics_are_synthetic": (
                    data_label.upper().startswith("SYNTH")
                ),
                "dataset_sha256": provenance.get(
                    "dataset_sha256"
                ),
                "split_steps": provenance.get(
                    "split_steps"
                ),
                **metrics,
            },
        }

        return {
            **base,
            "model": model_info,
        }