"""Model artifact saving/loading with integrity and compatibility checks.

Trust boundary:
- Artifacts must be produced by this project in a trusted environment.
- Joblib artifacts use pickle-based deserialization.
- SHA-256 detects model-file changes relative to metadata.json.
- SHA-256 alone does NOT authenticate metadata or protect against an
  attacker who can replace both metadata and the model file.
- Never load artifacts from untrusted sources.

The model file is verified before deserialization.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline.features import FeatureContract, build_features
from ml.pipeline.models import (
    MODEL_KINDS,
    ScoreModel,
    library_versions,
)
from ml.pipeline.schema import validate_schema


ARTIFACT_FORMAT_VERSION = 1
METADATA_FILE = "metadata.json"

SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")

REQUIRED_METADATA_KEYS = {
    "artifact_format_version",
    "created_utc",
    "model_kind",
    "model_file",
    "model_sha256",
    "seed",
    "feature_contract",
    "threshold",
    "metrics",
    "provenance",
    "library_versions",
    "score_semantics",
}

RESERVED_EXTRA_KEYS = REQUIRED_METADATA_KEYS | {
    "params",
    "fit_info",
    "reference_profile",
}


class ArtifactError(RuntimeError):
    """Missing, corrupted, tampered or incompatible model artifact."""


def _sha256(path: Path) -> str:
    """Calculate SHA-256 using bounded memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)

    return digest.hexdigest()


def _jsonable(value):
    """Convert supported NumPy values to JSON-compatible values."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        result = float(value)

        if not math.isfinite(result):
            raise ValueError("Non-finite float cannot be stored in metadata.")

        return result

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Not JSON serialisable: {type(value)}"
    )


def _validate_threshold_metadata(threshold: dict) -> float:
    """Validate the saved operating threshold."""

    if not isinstance(threshold, dict):
        raise ArtifactError("Threshold metadata must be a dictionary.")

    if "threshold" not in threshold:
        raise ArtifactError(
            "Threshold metadata is missing the 'threshold' value."
        )

    try:
        value = float(threshold["threshold"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "Saved threshold must be numeric."
        ) from exc

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ArtifactError(
            "Saved threshold must be finite and between 0 and 1."
        )

    return value


def _expected_model_filename(kind: str) -> str:
    """Return the expected native artifact filename for a model kind."""

    if kind == "xgboost":
        return "model.json"

    if kind in ("hist_gbm", "logreg"):
        return "model.joblib"

    raise ArtifactError(
        f"Unsupported model kind: {kind!r}"
    )


def _validate_metadata(meta: dict) -> None:
    """Validate metadata structure before using any metadata paths."""

    if not isinstance(meta, dict):
        raise ArtifactError(
            "metadata.json must contain a JSON object."
        )

    missing = REQUIRED_METADATA_KEYS - set(meta)

    if missing:
        raise ArtifactError(
            f"metadata.json is missing required keys: {sorted(missing)}"
        )

    if meta["artifact_format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(
            f"Unsupported artifact format "
            f"{meta['artifact_format_version']}; "
            f"this code reads format {ARTIFACT_FORMAT_VERSION}."
        )

    kind = meta["model_kind"]

    if kind not in MODEL_KINDS:
        raise ArtifactError(
            f"Unsupported model kind: {kind!r}."
        )

    if not isinstance(meta["model_file"], str):
        raise ArtifactError("model_file must be a string.")

    expected_filename = _expected_model_filename(kind)

    # Validate the basename only; never follow a metadata-supplied path.
    metadata_basename = meta["model_file"].replace("\\", "/").rsplit("/", 1)[-1]
    if metadata_basename != expected_filename:
        raise ArtifactError(
            f"Invalid model filename for {kind}: "
            f"expected '{expected_filename}'."
        )

    checksum = meta["model_sha256"]

    if (
        not isinstance(checksum, str)
        or not SHA256_PATTERN.fullmatch(checksum)
    ):
        raise ArtifactError(
            "metadata.json contains an invalid SHA-256 checksum."
        )

    if not isinstance(meta["feature_contract"], dict):
        raise ArtifactError(
            "feature_contract must be a dictionary."
        )

    if not isinstance(meta["library_versions"], dict):
        raise ArtifactError(
            "library_versions must be a dictionary."
        )

    if not isinstance(meta["threshold"], dict):
        raise ArtifactError(
            "threshold must be a dictionary."
        )

    if not isinstance(meta["metrics"], dict):
        raise ArtifactError(
            "metrics must be a dictionary."
        )

    if not isinstance(meta["provenance"], dict):
        raise ArtifactError(
            "provenance must be a dictionary."
        )

    try:
        _validate_threshold_metadata(meta["threshold"])
    except ArtifactError:
        raise

    try:
        int(meta["seed"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "Artifact seed must be an integer."
        ) from exc


def save_artifact(
    directory: Path,
    model: ScoreModel,
    contract: FeatureContract,
    *,
    threshold: dict,
    metrics: dict,
    provenance: dict,
    reference_profile: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Save a model, its feature contract and integrity metadata."""

    directory = Path(directory)

    if model.est is None:
        raise ArtifactError(
            "Cannot save an artifact from an unfitted model."
        )

    if not isinstance(contract, FeatureContract):
        raise ArtifactError(
            "contract must be a FeatureContract instance."
        )

    if model.kind not in MODEL_KINDS:
        raise ArtifactError(
            f"Unsupported model kind: {model.kind!r}"
        )

    _validate_threshold_metadata(threshold)

    if not isinstance(metrics, dict):
        raise ArtifactError("metrics must be a dictionary.")

    if not isinstance(provenance, dict):
        raise ArtifactError("provenance must be a dictionary.")

    if reference_profile is not None and not isinstance(
        reference_profile, dict
    ):
        raise ArtifactError(
            "reference_profile must be a dictionary or None."
        )

    if extra is not None and not isinstance(extra, dict):
        raise ArtifactError("extra must be a dictionary or None.")

    if extra:
        reserved = RESERVED_EXTRA_KEYS & set(extra)

        if reserved:
            raise ArtifactError(
                "extra metadata cannot override reserved fields: "
                f"{sorted(reserved)}"
            )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_filename = _expected_model_filename(model.kind)

    try:
        model_filename = model.save(directory)
    except Exception as exc:
        raise ArtifactError(
            f"Could not save model artifact: {exc}"
        ) from exc

    if model_filename != expected_filename:
        raise ArtifactError(
            f"Model save returned unexpected filename "
            f"{model_filename!r}; expected {expected_filename!r}."
        )

    model_path = directory / expected_filename

    if not model_path.is_file():
        raise ArtifactError(
            f"Model save did not create {expected_filename}."
        )

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "model_kind": model.kind,
        "seed": int(model.seed),
        "model_file": expected_filename,
        "model_sha256": _sha256(model_path),
        "params": getattr(model, "params_used", model.params),
        "fit_info": model.fit_info,
        "feature_contract": contract.to_dict(),
        "threshold": threshold,
        "metrics": metrics,
        "provenance": provenance,
        "library_versions": library_versions(),
        "score_semantics": (
            "Ranking score from a class-weighted model; "
            "NOT a calibrated probability."
        ),
    }

    if reference_profile is not None:
        metadata["reference_profile"] = reference_profile

    if extra:
        metadata.update(extra)

    metadata_path = directory / METADATA_FILE
    temporary_path = directory / f"{METADATA_FILE}.tmp"

    try:
        serialized = json.dumps(
            metadata,
            indent=2,
            default=_jsonable,
            allow_nan=False,
        )

        temporary_path.write_text(
            serialized,
            encoding="utf-8",
        )

        # Replace metadata only after successful serialization.
        temporary_path.replace(metadata_path)

    except (OSError, TypeError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)

        raise ArtifactError(
            f"Could not write artifact metadata: {exc}"
        ) from exc

    return metadata


@dataclass
class LoadedModel:
    """Verified model and its associated feature contract."""

    model: ScoreModel
    contract: FeatureContract
    metadata: dict

    @property
    def threshold(self) -> float:
        return _validate_threshold_metadata(
            self.metadata["threshold"]
        )

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Score raw transactions using the saved feature contract.

        Required raw columns depend on the saved feature set.
        The production base feature set does not require balance columns.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Prediction input must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Prediction input cannot be empty."
            )

        validate_schema(
            df,
            require_target=False,
        )

        features = build_features(
            df,
            self.contract,
        )

        return self.model.predict_score(features)

    def predict_label(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using the frozen validation threshold."""

        scores = self.predict_score(df)

        return (
            scores >= self.threshold
        ).astype(np.int32)


def _major(version: str) -> str:
    return str(version).split(".")[0]


def _minor(version: str) -> str:
    return ".".join(str(version).split(".")[:2])


def _check_library_compatibility(meta: dict) -> None:
    """Check model-library compatibility before deserializing."""

    saved = meta["library_versions"]
    current = library_versions()

    kind = meta["model_kind"]

    if kind == "xgboost":
        saved_version = saved.get("xgboost")
        current_version = current.get("xgboost")

        if not saved_version:
            raise ArtifactError(
                "Artifact metadata does not contain the XGBoost version."
            )

        if not current_version:
            raise ArtifactError(
                "Artifact needs xgboost, which is not installed."
            )

        if _major(saved_version) != _major(current_version):
            raise ArtifactError(
                f"xgboost major version mismatch: "
                f"trained {saved_version}, installed {current_version}."
            )

    else:
        saved_version = saved.get("sklearn")
        current_version = current.get("sklearn")

        if not saved_version:
            raise ArtifactError(
                "Artifact metadata does not contain the scikit-learn version."
            )

        if not current_version:
            raise ArtifactError(
                "scikit-learn is not installed."
            )

        if _minor(saved_version) != _minor(current_version):
            raise ArtifactError(
                f"scikit-learn version mismatch: "
                f"trained {saved_version}, installed {current_version}. "
                "Retrain in this environment."
            )


def load_artifact(
    directory: str | Path,
) -> LoadedModel:
    """Verify and load a locally trusted model artifact."""

    directory = Path(directory)

    metadata_path = directory / METADATA_FILE

    if not directory.is_dir() or not metadata_path.is_file():
        raise ArtifactError(
            f"No model artifact found at '{directory.name}' "
            f"(missing {METADATA_FILE}). "
            "Train a model first: "
            "python -m ml.train --data <csv>"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ArtifactError(
            "metadata.json is corrupted or unreadable."
        ) from exc

    _validate_metadata(metadata)

    # Only a known filename is allowed. No metadata-provided directory
    # traversal or absolute path can be followed.
    expected_filename = _expected_model_filename(
        metadata["model_kind"]
    )

    model_path = directory / expected_filename

    if not model_path.is_file():
        raise ArtifactError(
            f"Model file '{expected_filename}' is missing from the artifact."
        )

    try:
        actual_checksum = _sha256(model_path)
    except OSError as exc:
        raise ArtifactError(
            f"Could not read model artifact '{expected_filename}'."
        ) from exc

    if actual_checksum.lower() != metadata["model_sha256"].lower():
        raise ArtifactError(
            "Model file SHA-256 does not match metadata "
            "(modified or corrupted)."
        )

    # Compatibility is checked before model deserialization.
    _check_library_compatibility(metadata)

    try:
        contract = FeatureContract.from_dict(
            metadata["feature_contract"]
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactError(
            f"Incompatible feature contract: {exc}"
        ) from exc

    try:
        model = ScoreModel.load(
            metadata["model_kind"],
            model_path,
            seed=int(metadata.get("seed", 42)),
        )
    except Exception as exc:
        raise ArtifactError(
            f"Could not load model artifact: {exc}"
        ) from exc

    return LoadedModel(
        model=model,
        contract=contract,
        metadata=metadata,
    )