"""Model artifact saving/loading with integrity and compatibility checks.

Trust boundary: artifacts are produced locally by ml/train.py. Never load an artifact directory
that came from an untrusted source (joblib models are pickle-based). Loading verifies a SHA-256
of the model file against metadata.json before deserialising.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline.features import FeatureContract, build_features
from ml.pipeline.models import ScoreModel, library_versions
from ml.pipeline.schema import validate_schema

ARTIFACT_FORMAT_VERSION = 1
METADATA_FILE = "metadata.json"


class ArtifactError(RuntimeError):
    """Missing, corrupted, tampered or incompatible model artifact."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_artifact(directory: Path, model: ScoreModel, contract: FeatureContract, *,
                  threshold: dict, metrics: dict, provenance: dict,
                  reference_profile: dict | None = None, extra: dict | None = None) -> dict:
    model_file = model.save(directory)
    meta = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_kind": model.kind, "seed": model.seed, "model_file": model_file,
        "model_sha256": _sha256(directory / model_file),
        "params": getattr(model, "params_used", model.params), "fit_info": model.fit_info,
        "feature_contract": contract.to_dict(), "threshold": threshold, "metrics": metrics,
        "provenance": provenance, "library_versions": library_versions(),
        "score_semantics": "ranking score from a class-weighted model; NOT a calibrated probability",
    }
    if reference_profile is not None:
        meta["reference_profile"] = reference_profile
    if extra:
        meta.update(extra)
    (directory / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=_jsonable),
                                           encoding="utf-8")
    return meta


@dataclass
class LoadedModel:
    model: ScoreModel
    contract: FeatureContract
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"]["threshold"])

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Score raw transactions. Needs only step,type,amount,nameOrig,nameDest (no balances)."""
        validate_schema(df, require_target=False)
        return self.model.predict_score(build_features(df, self.contract))

    def predict_label(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_score(df) >= self.threshold).astype(int)


def _major(v: str) -> str:
    return v.split(".")[0]


def _minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def load_artifact(directory: str | Path) -> LoadedModel:
    d = Path(directory)
    meta_path = d / METADATA_FILE
    if not d.is_dir() or not meta_path.is_file():
        raise ArtifactError(f"No model artifact found at '{d.name}' (missing {METADATA_FILE}). "
                            "Train a model first: python -m ml.train --data <csv>")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ArtifactError("metadata.json is corrupted (invalid JSON).") from e
    for key in ("artifact_format_version", "model_kind", "model_file", "model_sha256",
                "feature_contract", "threshold", "library_versions"):
        if key not in meta:
            raise ArtifactError(f"metadata.json is missing required key '{key}'.")
    if meta["artifact_format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(f"Unsupported artifact format {meta['artifact_format_version']}; "
                            f"this code reads format {ARTIFACT_FORMAT_VERSION}.")
    model_path = d / Path(meta["model_file"]).name  # never follow paths from metadata
    if not model_path.is_file():
        raise ArtifactError(f"Model file '{model_path.name}' is missing from the artifact.")
    if _sha256(model_path) != meta["model_sha256"]:
        raise ArtifactError("Model file SHA-256 does not match metadata (modified or corrupted).")
    saved, now = meta["library_versions"], library_versions()
    if meta["model_kind"] == "xgboost":
        if "xgboost" not in now:
            raise ArtifactError("Artifact needs xgboost, which is not installed.")
        if _major(saved.get("xgboost", "0")) != _major(now["xgboost"]):
            raise ArtifactError(f"xgboost major version mismatch: trained {saved.get('xgboost')}, "
                                f"installed {now['xgboost']}.")
    else:
        if _minor(saved.get("sklearn", "0.0")) != _minor(now["sklearn"]):
            raise ArtifactError(f"scikit-learn version mismatch: trained {saved.get('sklearn')}, "
                                f"installed {now['sklearn']}. Retrain in this environment.")
    try:
        contract = FeatureContract.from_dict(meta["feature_contract"])
    except (ValueError, KeyError) as e:
        raise ArtifactError(f"Incompatible feature contract: {e}") from e
    model = ScoreModel.load(meta["model_kind"], model_path, seed=int(meta.get("seed", 42)))
    return LoadedModel(model=model, contract=contract, metadata=meta)
