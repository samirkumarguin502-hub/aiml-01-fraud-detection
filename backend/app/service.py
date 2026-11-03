"""Framework-independent service layer. The FastAPI routes only call these methods.

Prediction reuses ml.pipeline (schema validation + shared FeatureContract) via the loaded artifact.
History features: a sender's earlier transactions that were previously scored by THIS API are
prepended to the frame, so `orig_prev_*` mean "earlier steps seen by this platform", not the
full dataset history the model trained on. Documented as train/serve skew in docs/api.md.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import uuid

import pandas as pd

from backend.app.config import Settings
from backend.app.db import CASE_STATUSES, Repository
from backend.app.errors import AppError
from backend.app.logging_setup import log_event
from backend.app.model_service import ModelService
from backend.app.validation import TX_FIELDS, parse_csv, validate_transaction
from ml.pipeline.model_io import ArtifactError, load_artifact
from monitoring.drift import build_report

log = logging.getLogger("fraudapp")
SCORE_NOTE = "Uncalibrated ranking score from a class-weighted model; not a probability of fraud."
NOTE_MAX = 500


def _text(v, field: str, max_len: int = NOTE_MAX) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or len(v) > max_len or "\x00" in v:
        raise AppError(422, "validation_error", f"{field} must be text of at most {max_len} characters.")
    return v.strip() or None


class FraudService:
    def __init__(self, settings: Settings, repo: Repository, model: ModelService):
        self.settings, self.repo, self.model = settings, repo, model

    # ---- health ----
    def health(self) -> dict:
        return {"status": "alive"}

    def readiness(self) -> tuple[int, dict]:
        db_ok = self.repo.ping()
        checks = {"database": "ok" if db_ok else "unavailable", "model": self.model.state}
        ready = db_ok and self.model.ready
        body = {"ready": ready, "checks": checks}
        if not self.model.ready:
            body["model_reason"] = self.model.reason
        return (200 if ready else 503), body

    def model_info(self) -> dict:
        return self.model.info()

    # ---- prediction ----
    def _model_meta(self) -> dict:
        m = self.model.require_ready().metadata
        return {"kind": m["model_kind"], "sha256": m["model_sha256"],
                "data_label": m.get("provenance", {}).get("data_label")}

    def _fields(self) -> tuple:
        """Inputs the loaded model needs; the base five when no model is loaded (every path then returns 503)."""
        return tuple(self.model.loaded.contract.input_columns) if self.model.ready else TX_FIELDS

    def _score_new(self, new_df: pd.DataFrame) -> list[dict]:
        lm = self.model.require_ready()
        hist = self.repo.history_for(new_df["nameOrig"].tolist())
        cols = list(self._fields())
        if hist:
            h = pd.DataFrame(hist, columns=list(TX_FIELDS)).astype({"step": "int64", "amount": "float64"})
            for c in cols:
                if c not in h.columns:
                    h[c] = 0.0  # history rows only feed sender-history features; their balances are not used
            frame = pd.concat([h[cols], new_df[cols]], ignore_index=True)
        else:  # never concat an empty untyped frame: it degrades step/amount to object dtype
            frame = new_df[cols].reset_index(drop=True)
        scores = self.model.score(frame)[len(hist):]
        thr = lm.threshold
        vocab = set(lm.contract.type_vocab)
        out = []
        for rec, s in zip(new_df.to_dict("records"), scores):
            out.append({**rec, "score": float(s), "threshold": thr, "flagged": bool(s >= thr),
                        "type_known": rec["type"] in vocab})
        return out

    def _model_brief(self) -> dict:
        m = self._model_meta()
        return {"kind": m["kind"], "sha256": m["sha256"], "metrics_origin": m["data_label"]}

    def predict_one(self, payload) -> dict:
        try:
            tx = validate_transaction(payload, self._fields())
        except AppError as e:
            self.repo.record_rejection("predict", e.code)
            raise
        self.model.require_ready()
        (row,) = self._score_new(pd.DataFrame([tx]))
        (tid,) = self.repo.insert_predictions([row], "single", None, self._model_meta())
        log_event(log, "predict", flagged=row["flagged"], model_state=self.model.state)
        warnings = [] if row["type_known"] else [
            "Transaction type was not seen in training; it is scored in the 'unknown' bucket."]
        return {"transaction_id": tid, "score": row["score"], "threshold": row["threshold"],
                "flagged": row["flagged"], "score_note": SCORE_NOTE, "warnings": warnings,
                "model": self._model_brief()}

    def predict_batch(self, raw: bytes) -> dict:
        try:
            df = parse_csv(raw, self.settings.max_upload_bytes, self.settings.max_csv_rows, self._fields())
        except AppError as e:
            self.repo.record_rejection("predict_batch", e.code)
            raise
        self.model.require_ready()
        rows = self._score_new(df)
        batch_id = uuid.uuid4().hex
        ids = self.repo.insert_predictions(rows, "batch", batch_id, self._model_meta())
        flagged = sum(r["flagged"] for r in rows)
        log_event(log, "predict_batch", rows=len(rows), flagged=flagged, batch_id=batch_id)
        return {"batch_id": batch_id, "rows": len(rows), "flagged": flagged, "score_note": SCORE_NOTE,
                "unknown_type_rows": sum(not r["type_known"] for r in rows), "model": self._model_brief(),
                "results": [{"row": i + 2, "transaction_id": tid, "score": r["score"], "flagged": r["flagged"]}
                            for i, (tid, r) in enumerate(zip(ids, rows))]}

    # ---- history / cases ----
    def _page(self, page, page_size):
        if not isinstance(page, int) or not isinstance(page_size, int) or page < 1 \
                or not 1 <= page_size <= self.settings.max_page_size:
            raise AppError(422, "validation_error", "Invalid pagination.",
                           details={"page": ">= 1", "page_size": f"1..{self.settings.max_page_size}"})

    def list_transactions(self, *, page: int = 1, page_size: int = 25, **filters) -> dict:
        self._page(page, page_size)
        allowed = {"sort", "order", "flagged", "type_", "min_score", "max_score", "source",
                   "case_status", "batch_id"}
        if set(filters) - allowed:
            raise AppError(422, "validation_error", "Unsupported filter.")
        if filters.get("case_status") not in (None, *CASE_STATUSES):
            raise AppError(422, "validation_error", "Invalid case_status.", details={"allowed": list(CASE_STATUSES)})
        if filters.get("source") not in (None, "single", "batch"):
            raise AppError(422, "validation_error", "source must be 'single' or 'batch'.")
        return self.repo.list_transactions(page=page, page_size=page_size, **filters)

    def get_transaction(self, tid: int) -> dict:
        return self.repo.get_transaction(self._id(tid))

    def alerts(self, *, page: int = 1, page_size: int = 25, case_status: str | None = None) -> dict:
        r = self.list_transactions(page=page, page_size=page_size, flagged=True,
                                   case_status=case_status, sort="created_utc", order="desc")
        return {**r, "note": "Alerts are transactions whose score met the model's validation-selected threshold."}

    @staticmethod
    def _id(tid) -> int:
        if not isinstance(tid, int) or isinstance(tid, bool) or tid < 1:
            raise AppError(422, "validation_error", "Invalid transaction id.")
        return tid

    def update_case(self, tid: int, payload) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"status", "note"} or "status" not in payload:
            raise AppError(422, "validation_error", "Body must contain 'status' and optionally 'note'.")
        return self.repo.update_case(self._id(tid), payload["status"], _text(payload.get("note"), "note"))

    def set_feedback(self, tid: int, payload) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"label", "comment"} or "label" not in payload:
            raise AppError(422, "validation_error", "Body must contain 'label' (0 or 1) and optionally 'comment'.")
        lab = payload["label"]
        if isinstance(lab, bool) or lab not in (0, 1):
            raise AppError(422, "validation_error", "label must be 0 (not fraud) or 1 (fraud).")
        return self.repo.set_feedback(self._id(tid), int(lab), _text(payload.get("comment"), "comment"))

    # ---- dashboard ----
    def dashboard_summary(self) -> dict:
        s = self.repo.summary()
        info = self.model.info()
        return {"empty": s["total_transactions"] == 0, **s,
                "model": {"state": info["state"], "reason": info["reason"],
                          "kind": (info["model"] or {}).get("kind"),
                          "metrics_origin": ((info["model"] or {}).get("metrics") or {}).get("origin")},
                "notes": ["Counts come only from transactions scored through this API.",
                          "Scores are uncalibrated ranking scores, not probabilities.",
                          "Feedback counts are analyst labels on a non-random sample; do not read them as model accuracy."]}

    # ---- monitoring ----
    def monitoring_status(self) -> dict:
        s = self.repo.summary()
        volume = {"total_transactions": s["total_transactions"], "flagged_transactions": s["flagged_transactions"],
                  "flag_rate": (s["flagged_transactions"] / s["total_transactions"]) if s["total_transactions"] else None,
                  "by_day": s["by_day"], "score_histogram": s["score_histogram"]}
        rep = build_report(rows=self.repo.monitoring_rows(), volume=volume, rejections=self.repo.rejection_counts(),
                           labelled=self.repo.labelled_rows(), loaded_model=self.model.loaded,
                           model_state=self.model.state, min_rows=self.settings.min_monitor_rows)
        rep["active_model"] = self._model_brief() if self.model.ready else None
        return rep

    def pipeline_status(self) -> dict:
        def installed(mod: str) -> bool:
            try:
                return importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                return False
        comps = {"xgboost": "xgboost", "mlflow": "mlflow", "kafka_client (confluent-kafka)": "confluent_kafka",
                 "feast": "feast", "evidently": "evidently", "torch": "torch",
                 "torch_geometric": "torch_geometric"}
        last = None
        f = self.settings.outcomes_dir / "last_run_summary.json"
        try:
            if f.is_file() and f.stat().st_size < 100_000:
                last = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last = {"error": "unreadable"}
        return {"optional_components": {k: {"library_installed": installed(v)} for k, v in comps.items()},
                "note": ("'library_installed' only means the Python package can be imported. It does not mean a Kafka "
                         "broker, Feast store or MLflow server is running or has been verified."),
                "last_replay_summary": last,
                "model_state": self.model.state, "database": "ok" if self.repo.ping() else "unavailable"}

    # ---- retraining workflow ----
    def retrain_create(self, payload) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"reason", "note"} or "reason" not in payload:
            raise AppError(422, "validation_error", "Body must contain 'reason' and optionally 'note'.")
        reason = _text(payload["reason"], "reason")
        if not reason:
            raise AppError(422, "validation_error", "reason is required.")
        return self.repo.create_retrain(reason, _text(payload.get("note"), "note"))

    def retrain_list(self) -> dict:
        return {"items": self.repo.list_retrain(),
                "note": "Promotion here is a recorded decision only. Activating a model means pointing "
                        "MODEL_ARTIFACT_DIR at it and restarting the API."}

    def _candidate_meta(self, name) -> dict:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", name):
            raise AppError(422, "validation_error", "candidate must be an artifact run folder name.")
        try:
            return load_artifact(self.settings.artifact_root / name / "primary").metadata
        except ArtifactError as e:
            raise AppError(422, "candidate_invalid", f"Candidate artifact rejected: {e}") from None

    def retrain_transition(self, rid: int, payload) -> dict:
        self._id(rid)
        if not isinstance(payload, dict) or set(payload) - {"to", "note", "candidate"} or "to" not in payload:
            raise AppError(422, "validation_error", "Body must contain 'to' and optionally 'note', 'candidate'.")
        to, note = payload["to"], _text(payload.get("note"), "note")
        if to == "evaluate":  # the SERVER decides passed/failed; the client cannot assert a pass
            cur = self.repo.get_retrain(rid)
            if cur["status"] != "candidate_trained" or not cur["candidate"]:
                raise AppError(409, "invalid_transition", "Only a request with a trained candidate can be evaluated.")
            cand = self._candidate_meta(cur["candidate"])
            act = self.model.loaded.metadata if self.model.ready else None
            c_pr = (cand.get("metrics", {}).get("test") or {}).get("pr_auc")
            a_pr = ((act or {}).get("metrics", {}).get("test") or {}).get("pr_auc")
            passed = c_pr is not None and a_pr is not None and c_pr >= a_pr
            ev = {"rule": "candidate held-out test PR-AUC >= active model's held-out test PR-AUC",
                  "candidate_test_pr_auc": c_pr, "active_test_pr_auc": a_pr, "passed": passed,
                  "candidate_origin": cand.get("provenance", {}).get("data_label"),
                  "active_origin": (act or {}).get("provenance", {}).get("data_label") if act else None,
                  "caveat": "The two models were scored on their own held-out windows; this is a screening "
                            "gate, not proof of superiority. Review the reports before promoting."}
            return self.repo.transition_retrain(rid, "evaluated_passed" if passed else "evaluated_failed",
                                                note, evaluation=ev)
        if to == "candidate_trained":
            cand = payload.get("candidate")
            self._candidate_meta(cand)
            return self.repo.transition_retrain(rid, to, note, candidate=cand)
        if to in ("promoted", "rejected", "cancelled"):
            return self.repo.transition_retrain(rid, to, note)
        raise AppError(422, "validation_error", "Unsupported target status.",
                       details={"allowed": ["candidate_trained", "evaluate", "promoted", "rejected", "cancelled"]})
