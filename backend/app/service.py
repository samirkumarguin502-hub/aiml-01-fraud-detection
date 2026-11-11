"""Framework-independent service layer.

FastAPI routes call these methods. Business logic remains independent
of the HTTP framework.

Prediction reuses ml.pipeline schema validation and the loaded
artifact's FeatureContract.

History features:
A sender's earlier transactions previously scored by THIS API are
prepended to the frame. Therefore orig_prev_* means earlier steps
seen by this platform, not the complete training-dataset history.

This train/serve skew is documented in docs/api.md.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import re
import uuid

import pandas as pd

from backend.app.config import Settings
from backend.app.db import CASE_STATUSES, Repository
from backend.app.errors import AppError
from backend.app.logging_setup import log_event
from backend.app.model_service import ModelService
from backend.app.validation import (
    TX_FIELDS,
    parse_csv,
    validate_transaction,
)

from ml.pipeline.model_io import ArtifactError, load_artifact
from monitoring.drift import build_report


log = logging.getLogger("fraudapp")

SCORE_NOTE = (
    "Uncalibrated ranking score from a class-weighted model; "
    "not a probability of fraud."
)

NOTE_MAX = 500

CANDIDATE_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,80}$"
)


# ----------------------------------------------------------------------
# Shared validation helpers
# ----------------------------------------------------------------------

def _text(
    value,
    field: str,
    max_len: int = NOTE_MAX,
) -> str | None:
    """Validate and normalize optional user-provided text."""

    if value is None:
        return None

    if (
        not isinstance(value, str)
        or len(value) > max_len
        or "\x00" in value
    ):
        raise AppError(
            422,
            "validation_error",
            f"{field} must be text of at most {max_len} characters.",
        )

    return value.strip() or None


def _metric_value(metrics: dict, section: str) -> float | None:
    """Read and validate a PR-AUC value from artifact metadata."""

    if not isinstance(metrics, dict):
        return None

    section_data = metrics.get(section)

    if not isinstance(section_data, dict):
        return None

    value = section_data.get("pr_auc")

    if isinstance(value, bool):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None

    return value


# ----------------------------------------------------------------------
# FraudService
# ----------------------------------------------------------------------

class FraudService:
    def __init__(
        self,
        settings: Settings,
        repo: Repository,
        model: ModelService,
    ):
        self.settings = settings
        self.repo = repo
        self.model = model

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return {
            "status": "alive",
        }

    def readiness(self) -> tuple[int, dict]:
        db_ok = self.repo.ping()

        checks = {
            "database": "ok" if db_ok else "unavailable",
            "model": self.model.state,
        }

        ready = db_ok and self.model.ready

        body = {
            "ready": ready,
            "checks": checks,
        }

        if not self.model.ready:
            body["model_reason"] = self.model.reason

        return (
            200 if ready else 503,
            body,
        )

    def model_info(self) -> dict:
        return self.model.info()

    # ------------------------------------------------------------------
    # Model metadata and input contract
    # ------------------------------------------------------------------

    def _model_meta(self) -> dict:
        metadata = self.model.require_ready().metadata

        provenance = metadata.get("provenance") or {}

        return {
            "kind": metadata["model_kind"],
            "sha256": metadata["model_sha256"],
            "data_label": provenance.get("data_label"),
        }

    def _fields(self) -> tuple:
        """Return the input fields required by the loaded model.

        The base transaction fields are returned when the model is
        unavailable. Prediction methods still enforce readiness.
        """

        if self.model.ready:
            return tuple(
                self.model.loaded.contract.input_columns
            )

        return TX_FIELDS

    def _model_brief(self) -> dict:
        metadata = self._model_meta()

        return {
            "kind": metadata["kind"],
            "sha256": metadata["sha256"],
            "metrics_origin": metadata["data_label"],
        }

    # ------------------------------------------------------------------
    # Prediction scoring
    # ------------------------------------------------------------------

    def _score_new(
        self,
        new_df: pd.DataFrame,
    ) -> list[dict]:
        """Score new transactions using available sender history."""

        if not isinstance(new_df, pd.DataFrame) or new_df.empty:
            raise AppError(
                422,
                "validation_error",
                "At least one valid transaction is required.",
            )

        required = set(TX_FIELDS)

        if not required.issubset(new_df.columns):
            raise AppError(
                422,
                "validation_error",
                "Transaction data is missing required fields.",
            )

        loaded_model = self.model.require_ready()

        history = self.repo.history_for(
            new_df["nameOrig"].tolist()
        )

        columns = list(self._fields())

        if not columns:
            raise AppError(
                503,
                "model_unavailable",
                "The model input contract is unavailable.",
            )

        missing_new = [
            column
            for column in columns
            if column not in new_df.columns
        ]

        if missing_new:
            raise AppError(
                422,
                "validation_error",
                "Transaction data does not satisfy the active model input contract.",
            )

        if history:
            history_df = pd.DataFrame(
                history,
                columns=list(TX_FIELDS),
            )

            history_df = history_df.astype(
                {
                    "step": "int64",
                    "amount": "float64",
                }
            )

            # Historical API records contain transaction fields only.
            # Balance fields are not available in the history table.
            for column in columns:
                if column not in history_df.columns:
                    history_df[column] = 0.0

            frame = pd.concat(
                [
                    history_df[columns],
                    new_df[columns],
                ],
                ignore_index=True,
            )

        else:
            # Avoid concatenating an empty untyped DataFrame.
            frame = new_df[columns].reset_index(drop=True)

        all_scores = self.model.score(frame)

        if len(all_scores) != len(frame):
            raise AppError(
                500,
                "prediction_error",
                "The model returned an inconsistent number of scores.",
            )

        scores = all_scores[len(history):]

        if len(scores) != len(new_df):
            raise AppError(
                500,
                "prediction_error",
                "The model returned an inconsistent prediction result.",
            )

        threshold = loaded_model.threshold

        vocabulary = set(
            loaded_model.contract.type_vocab
        )

        results = []

        for record, score in zip(
            new_df.to_dict("records"),
            scores,
        ):
            score = float(score)

            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise AppError(
                    500,
                    "prediction_error",
                    "The model returned an invalid ranking score.",
                )

            results.append(
                {
                    **record,
                    "score": score,
                    "threshold": threshold,
                    "flagged": bool(score >= threshold),
                    "type_known": record["type"] in vocabulary,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Single transaction prediction
    # ------------------------------------------------------------------

    def predict_one(self, payload) -> dict:
        try:
            transaction = validate_transaction(
                payload,
                self._fields(),
            )

        except AppError as exc:
            self.repo.record_rejection(
                "predict",
                exc.code,
            )
            raise

        self.model.require_ready()

        (row,) = self._score_new(
            pd.DataFrame([transaction])
        )

        (transaction_id,) = self.repo.insert_predictions(
            [row],
            "single",
            None,
            self._model_meta(),
        )

        log_event(
            log,
            "predict",
            flagged=row["flagged"],
            model_state=self.model.state,
        )

        warnings = []

        if not row["type_known"]:
            warnings.append(
                "Transaction type was not seen in training; "
                "it is scored in the 'unknown' bucket."
            )

        return {
            "transaction_id": transaction_id,
            "score": row["score"],
            "threshold": row["threshold"],
            "flagged": row["flagged"],
            "score_note": SCORE_NOTE,
            "warnings": warnings,
            "model": self._model_brief(),
        }

    # ------------------------------------------------------------------
    # Batch CSV prediction
    # ------------------------------------------------------------------

    def predict_batch(self, raw: bytes) -> dict:
        try:
            dataframe = parse_csv(
                raw,
                self.settings.max_upload_bytes,
                self.settings.max_csv_rows,
                self._fields(),
            )

        except AppError as exc:
            self.repo.record_rejection(
                "predict_batch",
                exc.code,
            )
            raise

        self.model.require_ready()

        rows = self._score_new(dataframe)

        batch_id = uuid.uuid4().hex

        transaction_ids = self.repo.insert_predictions(
            rows,
            "batch",
            batch_id,
            self._model_meta(),
        )

        if len(transaction_ids) != len(rows):
            raise AppError(
                500,
                "persistence_error",
                "The batch prediction results could not be stored consistently.",
            )

        flagged_count = sum(
            row["flagged"]
            for row in rows
        )

        log_event(
            log,
            "predict_batch",
            rows=len(rows),
            flagged=flagged_count,
            batch_id=batch_id,
        )

        return {
            "batch_id": batch_id,
            "rows": len(rows),
            "flagged": flagged_count,
            "score_note": SCORE_NOTE,
            "unknown_type_rows": sum(
                not row["type_known"]
                for row in rows
            ),
            "model": self._model_brief(),
            "results": [
                {
                    "row": index + 2,
                    "transaction_id": transaction_id,
                    "score": row["score"],
                    "flagged": row["flagged"],
                }
                for index, (transaction_id, row) in enumerate(
                    zip(transaction_ids, rows)
                )
            ],
        }

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def _page(
        self,
        page,
        page_size,
    ) -> None:
        """Validate pagination values strictly."""

        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or page < 1
            or not 1 <= page_size <= self.settings.max_page_size
        ):
            raise AppError(
                422,
                "validation_error",
                "Invalid pagination.",
                details={
                    "page": ">= 1",
                    "page_size": (
                        f"1..{self.settings.max_page_size}"
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def list_transactions(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        **filters,
    ) -> dict:
        self._page(page, page_size)

        allowed = {
            "sort",
            "order",
            "flagged",
            "type_",
            "min_score",
            "max_score",
            "source",
            "case_status",
            "batch_id",
        }

        if set(filters) - allowed:
            raise AppError(
                422,
                "validation_error",
                "Unsupported filter.",
            )

        case_status = filters.get("case_status")

        if (
            case_status is not None
            and case_status not in CASE_STATUSES
        ):
            raise AppError(
                422,
                "validation_error",
                "Invalid case_status.",
                details={
                    "allowed": list(CASE_STATUSES),
                },
            )

        source = filters.get("source")

        if source not in (None, "single", "batch"):
            raise AppError(
                422,
                "validation_error",
                "source must be 'single' or 'batch'.",
            )

        return self.repo.list_transactions(
            page=page,
            page_size=page_size,
            **filters,
        )

    def get_transaction(self, tid: int) -> dict:
        return self.repo.get_transaction(
            self._id(tid)
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def alerts(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        case_status: str | None = None,
    ) -> dict:
        result = self.list_transactions(
            page=page,
            page_size=page_size,
            flagged=True,
            case_status=case_status,
            sort="created_utc",
            order="desc",
        )

        return {
            **result,
            "note": (
                "Alerts are transactions whose score met the model's "
                "validation-selected threshold."
            ),
        }

    # ------------------------------------------------------------------
    # Transaction ID validation
    # ------------------------------------------------------------------

    @staticmethod
    def _id(tid) -> int:
        if (
            not isinstance(tid, int)
            or isinstance(tid, bool)
            or tid < 1
        ):
            raise AppError(
                422,
                "validation_error",
                "Invalid transaction id.",
            )

        return tid

    # ------------------------------------------------------------------
    # Analyst case management
    # ------------------------------------------------------------------

    def update_case(
        self,
        tid: int,
        payload,
    ) -> dict:
        if (
            not isinstance(payload, dict)
            or set(payload) - {"status", "note"}
            or "status" not in payload
        ):
            raise AppError(
                422,
                "validation_error",
                "Body must contain 'status' and optionally 'note'.",
            )

        return self.repo.update_case(
            self._id(tid),
            payload["status"],
            _text(
                payload.get("note"),
                "note",
            ),
        )

    # ------------------------------------------------------------------
    # Analyst feedback
    # ------------------------------------------------------------------

    def set_feedback(
        self,
        tid: int,
        payload,
    ) -> dict:
        if (
            not isinstance(payload, dict)
            or set(payload) - {"label", "comment"}
            or "label" not in payload
        ):
            raise AppError(
                422,
                "validation_error",
                "Body must contain 'label' (0 or 1) and optionally 'comment'.",
            )

        label = payload["label"]

        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            raise AppError(
                422,
                "validation_error",
                "label must be integer 0 or 1.",
            )

        return self.repo.set_feedback(
            self._id(tid),
            int(label),
            _text(
                payload.get("comment"),
                "comment",
            ),
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def dashboard_summary(self) -> dict:
        summary = self.repo.summary()
        info = self.model.info()

        model_metadata = info.get("model") or {}
        model_metrics = model_metadata.get("metrics") or {}

        return {
            "empty": summary["total_transactions"] == 0,
            **summary,
            "model": {
                "state": info["state"],
                "reason": info["reason"],
                "kind": model_metadata.get("kind"),
                "metrics_origin": model_metrics.get("origin"),
            },
            "notes": [
                "Counts come only from transactions scored through this API.",
                "Scores are uncalibrated ranking scores, not probabilities.",
                (
                    "Feedback counts are analyst labels on a non-random sample; "
                    "do not read them as model accuracy."
                ),
            ],
        }

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitoring_status(self) -> dict:
        summary = self.repo.summary()

        total = summary["total_transactions"]
        flagged = summary["flagged_transactions"]

        volume = {
            "total_transactions": total,
            "flagged_transactions": flagged,
            "flag_rate": (
                flagged / total
                if total
                else None
            ),
            "by_day": summary["by_day"],
            "score_histogram": summary["score_histogram"],
        }

        report = build_report(
            rows=self.repo.monitoring_rows(),
            volume=volume,
            rejections=self.repo.rejection_counts(),
            labelled=self.repo.labelled_rows(),
            loaded_model=self.model.loaded,
            model_state=self.model.state,
            min_rows=self.settings.min_monitor_rows,
        )

        report["active_model"] = (
            self._model_brief()
            if self.model.ready
            else None
        )

        return report

    # ------------------------------------------------------------------
    # Pipeline component status
    # ------------------------------------------------------------------

    def pipeline_status(self) -> dict:
        """Report optional component installation and replay summary."""

        def installed(module_name: str) -> bool:
            try:
                return (
                    importlib.util.find_spec(module_name)
                    is not None
                )
            except (ImportError, ValueError, ModuleNotFoundError):
                return False

        components = {
            "xgboost": "xgboost",
            "mlflow": "mlflow",
            "kafka_client (confluent-kafka)": "confluent_kafka",
            "feast": "feast",
            "evidently": "evidently",
            "torch": "torch",
            "torch_geometric": "torch_geometric",
        }

        last_summary = None

        summary_path = (
            self.settings.outcomes_dir
            / "last_run_summary.json"
        )

        try:
            if (
                summary_path.is_file()
                and summary_path.stat().st_size < 100_000
            ):
                last_summary = json.loads(
                    summary_path.read_text(
                        encoding="utf-8"
                    )
                )

                if not isinstance(last_summary, dict):
                    last_summary = {
                        "error": "invalid_summary_format",
                    }

        except (OSError, ValueError, TypeError):
            last_summary = {
                "error": "unreadable",
            }

        return {
            "optional_components": {
                name: {
                    "library_installed": installed(module_name),
                }
                for name, module_name in components.items()
            },
            "note": (
                "'library_installed' only means the Python package can be "
                "imported. It does not mean a Kafka broker, Feast store or "
                "MLflow server is running or has been verified."
            ),
            "last_replay_summary": last_summary,
            "model_state": self.model.state,
            "database": (
                "ok"
                if self.repo.ping()
                else "unavailable"
            ),
        }

    # ------------------------------------------------------------------
    # Retraining workflow
    # ------------------------------------------------------------------

    def retrain_create(self, payload) -> dict:
        if (
            not isinstance(payload, dict)
            or set(payload) - {"reason", "note"}
            or "reason" not in payload
        ):
            raise AppError(
                422,
                "validation_error",
                "Body must contain 'reason' and optionally 'note'.",
            )

        reason = _text(
            payload["reason"],
            "reason",
        )

        if not reason:
            raise AppError(
                422,
                "validation_error",
                "reason is required.",
            )

        return self.repo.create_retrain(
            reason,
            _text(
                payload.get("note"),
                "note",
            ),
        )

    def retrain_list(self) -> dict:
        return {
            "items": self.repo.list_retrain(),
            "note": (
                "Promotion here is a recorded decision only. Activating "
                "a model means pointing MODEL_ARTIFACT_DIR at it and "
                "restarting the API."
            ),
        }

    # ------------------------------------------------------------------
    # Candidate artifact loading
    # ------------------------------------------------------------------

    def _candidate_meta(self, name) -> dict:
        """Load a candidate artifact without exposing internal paths."""

        if (
            not isinstance(name, str)
            or not CANDIDATE_PATTERN.fullmatch(name)
        ):
            raise AppError(
                422,
                "validation_error",
                "candidate must be an artifact run folder name.",
            )

        artifact_path = (
            self.settings.artifact_root
            / name
            / "primary"
        )

        try:
            artifact = load_artifact(artifact_path)

            metadata = artifact.metadata

            if not isinstance(metadata, dict):
                raise ArtifactError(
                    "Invalid candidate metadata."
                )

            return metadata

        except AppError:
            raise

        except Exception as exc:
            # Keep paths and internal artifact details out of API responses.
            log_event(
                log,
                "candidate_artifact_rejected",
                logging.WARNING,
                exc_type=type(exc).__name__,
            )

            raise AppError(
                422,
                "candidate_invalid",
                "Candidate artifact is invalid or unavailable.",
            ) from None

    # ------------------------------------------------------------------
    # Retraining state transitions
    # ------------------------------------------------------------------

    def retrain_transition(
        self,
        rid: int,
        payload,
    ) -> dict:
        self._id(rid)

        if (
            not isinstance(payload, dict)
            or set(payload) - {"to", "note", "candidate"}
            or "to" not in payload
        ):
            raise AppError(
                422,
                "validation_error",
                "Body must contain 'to' and optionally 'note', 'candidate'.",
            )

        target_status = payload["to"]

        note = _text(
            payload.get("note"),
            "note",
        )

        # --------------------------------------------------------------
        # Candidate evaluation
        # --------------------------------------------------------------

        if target_status == "evaluate":
            current = self.repo.get_retrain(rid)

            if (
                current["status"] != "candidate_trained"
                or not current["candidate"]
            ):
                raise AppError(
                    409,
                    "invalid_transition",
                    (
                        "Only a request with a trained candidate "
                        "can be evaluated."
                    ),
                )

            candidate = self._candidate_meta(
                current["candidate"]
            )

            active = (
                self.model.loaded.metadata
                if self.model.ready
                else None
            )

            candidate_metrics = candidate.get("metrics") or {}
            active_metrics = (
                active.get("metrics") or {}
                if active
                else {}
            )

            candidate_pr_auc = _metric_value(
                candidate_metrics,
                "validation",
            )

            active_pr_auc = _metric_value(
                active_metrics,
                "validation",
            )

            # Do not compare models if validation metrics are missing.
            # Do not use held-out test metrics for promotion decisions.
            passed = (
                candidate_pr_auc is not None
                and active_pr_auc is not None
                and candidate_pr_auc >= active_pr_auc
            )

            evaluation = {
                "rule": (
                    "candidate validation PR-AUC >= active model "
                    "validation PR-AUC"
                ),
                "candidate_validation_pr_auc": candidate_pr_auc,
                "active_validation_pr_auc": active_pr_auc,
                "passed": passed,
                "candidate_origin": (
                    (candidate.get("provenance") or {}).get("data_label")
                ),
                "active_origin": (
                    (active.get("provenance") or {}).get("data_label")
                    if active
                    else None
                ),
                "caveat": (
                    "Models may use different validation windows. This is "
                    "a screening gate, not proof of superiority. Evaluate "
                    "both models on a common held-out dataset before promotion."
                ),
            }

            return self.repo.transition_retrain(
                rid,
                (
                    "evaluated_passed"
                    if passed
                    else "evaluated_failed"
                ),
                note,
                evaluation=evaluation,
            )

        # --------------------------------------------------------------
        # Register a trained candidate
        # --------------------------------------------------------------

        if target_status == "candidate_trained":
            candidate_name = payload.get("candidate")

            if not candidate_name:
                raise AppError(
                    422,
                    "validation_error",
                    "A candidate artifact name is required.",
                )

            self._candidate_meta(candidate_name)

            return self.repo.transition_retrain(
                rid,
                target_status,
                note,
                candidate=candidate_name,
            )

        # --------------------------------------------------------------
        # Other state transitions
        # Repository remains responsible for transition legality.
        # --------------------------------------------------------------

        if target_status in (
            "promoted",
            "rejected",
            "cancelled",
        ):
            return self.repo.transition_retrain(
                rid,
                target_status,
                note,
            )

        raise AppError(
            422,
            "validation_error",
            "Unsupported target status.",
            details={
                "allowed": [
                    "candidate_trained",
                    "evaluate",
                    "promoted",
                    "rejected",
                    "cancelled",
                ],
            },
        )