"""Monitoring + retraining workflow tests. SYNTHETIC data only; results say nothing about real drift."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises  # noqa: E402
from synthetic import make_synthetic  # noqa: E402
from test_backend_core import (META, GOOD, code_of, csv_bytes, make_service, ready_service,  # noqa: E402
                               row, trained_primary)

from backend.app.errors import AppError  # noqa: E402
from ml.pipeline.reference import psi, psi_level  # noqa: E402
from ml.train import run_training  # noqa: E402
from monitoring.drift import performance_report  # noqa: E402


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) < 1e-9
    assert psi([0.9, 0.1], [0.1, 0.9]) > 0.25 and psi_level(0.3) == "alert" and psi_level(0.15) == "warning"
    assert psi_level(0.01) == "stable"


def test_artifact_contains_reference_profile_without_raw_rows():
    import json
    meta = json.loads((trained_primary() / "metadata.json").read_text())
    prof = meta["reference_profile"]
    assert "day_index" not in prof["features"] and "amount" in prof["features"]
    assert abs(sum(prof["score"]["props"]) - 1) < 1e-9 and "rows" not in json.dumps(prof)[:0]


def test_no_model_state(tmp_path):
    r = make_service(tmp_path, None).monitoring_status()
    assert r["status"] == "no_model" and r["drift"] is None and r["drift_alert"] is False
    assert r["volume"]["total_transactions"] == 0 and r["volume"]["flag_rate"] is None


def test_insufficient_data_with_model_but_few_rows(tmp_path):
    svc = ready_service(tmp_path)
    svc.predict_one(GOOD)
    r = svc.monitoring_status()
    assert r["status"] == "insufficient_data" and r["drift"]["features"] is None if "features" in r["drift"] else True
    assert "features" not in r["drift"] and r["drift_alert"] is False and r["volume"]["total_transactions"] == 1


def test_similar_data_is_stable_and_shifted_data_alerts(tmp_path):
    svc = ready_service(tmp_path, min_monitor_rows=100)
    svc.predict_batch(csv_bytes(make_synthetic(60, 40, seed=0).head(2000)))
    stable = svc.monitoring_status()
    assert stable["status"] == "ok" and stable["drift"]["score"]["psi"] < 0.25
    assert not any(f["level"] == "alert" for f in stable["drift"]["features"] if f["feature"] != "orig_prev_txn_count"), stable["drift"]["features"]
    # shifted: huge amounts, only PAYMENT type
    shifted = make_synthetic(10, 40, seed=5).assign(type="PAYMENT", amount=5e8, nameDest="M1")
    svc2 = ready_service(tmp_path / "s" if (tmp_path / "s").mkdir() is None else tmp_path)
    svc2.predict_batch(csv_bytes(shifted))
    r = svc2.monitoring_status()
    assert r["status"] == "ok" and r["drift_alert"] is True and r["alert_reasons"]
    assert r["drift"]["features"][0]["level"] == "alert"


def test_drift_alert_does_not_change_model_or_create_requests(tmp_path):
    svc = ready_service(tmp_path)
    sha = svc.model.loaded.metadata["model_sha256"]
    svc.predict_batch(csv_bytes(make_synthetic(10, 40, seed=5).assign(type="PAYMENT", amount=5e8, nameDest="M1")))
    r = svc.monitoring_status()
    assert r["drift_alert"] is True and "does not retrain or replace" in r["retraining_note"]
    assert svc.model.loaded.metadata["model_sha256"] == sha and svc.model.state == "ready"
    assert svc.retrain_list()["items"] == []  # an alert never auto-creates a retraining request


def test_rejections_counted_by_code_without_content(tmp_path):
    svc = ready_service(tmp_path)
    for bad in ({**GOOD, "amount": -1}, {**GOOD, "amount": -2}, {"step": 1}):
        assert_raises(AppError, svc.predict_one, bad)
    assert_raises(AppError, svc.predict_batch, b"x,y\n1,2\n")
    q = svc.monitoring_status()["data_quality"]
    assert q["rejected_requests_total"] == 4 and q["rejected_requests_by_code"]["validation_error"] == 3
    assert q["rejected_requests_by_code"]["missing_columns"] == 1


def test_performance_insufficient_then_ok_and_labelled_as_biased():
    few = pd.DataFrame({"score": [0.9, 0.1], "threshold": 0.5, "label": [1, 0]})
    assert performance_report(few)["status"] == "insufficient_labels"
    one_class = pd.DataFrame({"score": np.linspace(0, 1, 30), "threshold": 0.5, "label": 0})
    assert performance_report(one_class)["status"] == "insufficient_labels"
    ok = pd.DataFrame({"score": [0.9, 0.8, 0.2, 0.1] * 10, "threshold": 0.5, "label": [1, 1, 0, 0] * 10})
    r = performance_report(ok)
    assert r["status"] == "ok" and r["precision"] == 1.0 and "biased sample" in r["warning"]


def test_performance_from_feedback_in_service(tmp_path):
    svc = ready_service(tmp_path)
    out = svc.predict_batch(csv_bytes(make_synthetic(10, 20)))
    for i, r in enumerate(out["results"][:30]):
        svc.set_feedback(r["transaction_id"], {"label": i % 2})
    p = svc.monitoring_status()["performance"]
    assert p["status"] == "ok" and p["labelled"] == 30


# ---------------- retraining workflow ----------------
def second_candidate(tmp_path) -> Path:
    """A second SYNTHETIC-trained artifact acting as the retrained candidate."""
    run_training(make_synthetic(seed=3), kind="hist_gbm", params={"max_iter": 40}, out_root=tmp_path,
                 run_id="cand1", data_label="SYNTHETIC test fixture", use_mlflow=False)
    return tmp_path


def test_retrain_state_machine_enforces_order(tmp_path):
    root = second_candidate(tmp_path / "arts" if (tmp_path / "arts").mkdir() is None else tmp_path)
    svc = ready_service(tmp_path, artifact_root=root)
    req = svc.retrain_create({"reason": "drift alert on amount", "note": "n"})
    rid = req["id"]
    assert req["status"] == "requested" and req["allowed_next"] == ["candidate_trained", "cancelled"]
    for bad in ({"to": "promoted"}, {"to": "evaluate"}, {"to": "evaluated_passed"}):
        e = assert_raises(AppError, svc.retrain_transition, rid, bad)
        assert e.code in ("invalid_transition", "validation_error")
    assert code_of(assert_raises(AppError, svc.retrain_transition, rid,
                                 {"to": "candidate_trained", "candidate": "nope"})) == "candidate_invalid"
    assert code_of(assert_raises(AppError, svc.retrain_transition, rid,
                                 {"to": "candidate_trained", "candidate": "../x"})) == "validation_error"
    r = svc.retrain_transition(rid, {"to": "candidate_trained", "candidate": "cand1"})
    assert r["status"] == "candidate_trained" and r["candidate"] == "cand1"
    assert assert_raises(AppError, svc.retrain_transition, rid, {"to": "promoted"}).status == 409
    r = svc.retrain_transition(rid, {"to": "evaluate"})
    assert r["status"] in ("evaluated_passed", "evaluated_failed")
    ev = r["evaluation"]
    assert ev["passed"] == (r["status"] == "evaluated_passed") and "caveat" in ev
    assert ev["candidate_origin"] == "SYNTHETIC test fixture"
    if r["status"] == "evaluated_passed":
        r = svc.retrain_transition(rid, {"to": "promoted", "note": "approved"})
        assert r["status"] == "promoted" and r["allowed_next"] == []
        assert svc.model.loaded.metadata["model_sha256"]  # active model untouched by promotion record
    else:
        assert svc.retrain_transition(rid, {"to": "rejected"})["status"] == "rejected"
    assert [e["to_status"] for e in svc.repo.get_retrain(rid)["events"]][0:2] == ["requested", "candidate_trained"]


def test_retrain_client_cannot_assert_a_pass(tmp_path):
    svc = ready_service(tmp_path)
    rid = svc.retrain_create({"reason": "r"})["id"]
    for to in ("evaluated_passed", "evaluated_failed", "requested", "bogus"):
        assert code_of(assert_raises(AppError, svc.retrain_transition, rid, {"to": to})) == "validation_error"


def test_retrain_validation_and_terminal_states(tmp_path):
    svc = make_service(tmp_path, None)
    for bad in ({}, {"reason": ""}, {"reason": "x", "extra": 1}, {"reason": "x" * 501}, "str"):
        assert code_of(assert_raises(AppError, svc.retrain_create, bad)) == "validation_error"
    rid = svc.retrain_create({"reason": "r"})["id"]
    assert svc.retrain_transition(rid, {"to": "cancelled"})["status"] == "cancelled"
    assert assert_raises(AppError, svc.retrain_transition, rid, {"to": "cancelled"}).status == 409
    assert assert_raises(AppError, svc.retrain_transition, 999, {"to": "cancelled"}).status == 404


def test_pipeline_status_reports_libraries_not_infrastructure(tmp_path):
    p = make_service(tmp_path, None).pipeline_status()
    assert "library_installed" in p["optional_components"]["feast"] and p["last_replay_summary"] is None
    assert "does not mean a Kafka broker" in p["note"]
    assert all(set(v) == {"library_installed"} for v in p["optional_components"].values())


def test_monitoring_cli_status_and_error_paths(tmp_path):
    import io
    import json as _json
    from contextlib import redirect_stdout
    from monitoring.__main__ import main
    db = tmp_path / "m.db"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["status", "--db", str(db)])
    out = _json.loads(buf.getvalue())
    assert code == 0 and out["status"] == "no_model" and out["volume"]["total_transactions"] == 0
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        code2 = main(["status", "--db", str(db), "--artifact", str(trained_primary())])
    out2 = _json.loads(buf2.getvalue())
    assert code2 == 0 and out2["status"] == "insufficient_data" and out2["active_model"]["kind"] == "hist_gbm"
    assert main(["evidently", "--db", str(db)]) == 2  # needs --data and a model
