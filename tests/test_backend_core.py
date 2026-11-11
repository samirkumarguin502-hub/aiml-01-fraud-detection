"""Backend core tests (no FastAPI needed). All data is SYNTHETIC; nothing here is a real result.
The 'ready model' tests use a real artifact trained by ml.train.run_training on tests/synthetic.py data."""
import io
import json
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.db import Repository  # noqa: E402
from backend.app.errors import AppError, ConfigError  # noqa: E402
from backend.app.model_service import ModelService  # noqa: E402
from backend.app.ratelimit import RateLimiter  # noqa: E402
from backend.app.service import FraudService  # noqa: E402
from backend.app.validation import parse_csv, validate_transaction  # noqa: E402
from ml.train import run_training  # noqa: E402

GOOD = dict(step=5, type="TRANSFER", amount=1234.5, nameOrig="C1", nameDest="C2")
_CACHE: dict = {}


def trained_primary() -> Path:
    """Train once per test process on SYNTHETIC data (hist_gbm, tiny)."""
    if "dir" not in _CACHE:
        root = Path(tempfile.mkdtemp())
        run_training(make_synthetic(), kind="hist_gbm", params={"max_iter": 40}, out_root=root,
                     run_id="fx", data_label="SYNTHETIC test fixture", use_mlflow=False)
        _CACHE["dir"] = root / "fx" / "primary"
    return _CACHE["dir"]


def make_service(tmp_path, model_dir=None, **over) -> FraudService:
    s = Settings(database_path=tmp_path / "t.db", model_dir=model_dir,
                 outcomes_dir=tmp_path / "streaming_outcomes", **over)
    repo = Repository(s.database_path)
    repo.init_schema()
    m = ModelService(s.model_dir)
    m.load()
    return FraudService(s, repo, m)


def ready_service(tmp_path, **over) -> FraudService:
    svc = make_service(tmp_path, trained_primary(), **over)
    assert svc.model.ready, svc.model.reason
    return svc


def code_of(exc: AppError) -> str:
    return exc.code


# ---------------- config ----------------
def test_settings_defaults_are_restrictive():
    s = Settings.from_env({})
    assert "*" not in s.cors_origins and s.trust_proxy is False and s.app_env == "development"
    assert s.database_path == Path("fraud_platform.db") and s.model_dir is None


def test_settings_reject_bad_values():
    for env in ({"CORS_ALLOW_ORIGINS": "*"}, {"MAX_CSV_ROWS": "abc"}, {"MAX_CSV_ROWS": "0"},
                {"DATABASE_URL": "postgresql://u:p@h/db"}, {"APP_ENV": "staging"}):
        assert_raises(ConfigError, Settings.from_env, env)


def test_settings_from_env_values():
    s = Settings.from_env({"APP_ENV": "production", "API_KEY": "test-only-key", "CORS_ALLOW_ORIGINS": "https://a.example, https://b.example",
                           "MODEL_ARTIFACT_DIR": "x/y", "MAX_CSV_ROWS": "10", "DATABASE_URL": "sqlite:///./z.db"})
    assert s.is_production and s.cors_origins == ("https://a.example", "https://b.example")
    assert s.max_csv_rows == 10 and s.model_dir == Path("x/y")


# ---------------- validation ----------------
def test_valid_transaction_normalised():
    assert validate_transaction({**GOOD, "step": 5.0, "amount": 3}) == {
        "step": 5, "type": "TRANSFER", "amount": 3.0, "nameOrig": "C1", "nameDest": "C2"}


def test_missing_none_and_extra_fields_rejected():
    for bad in ({k: v for k, v in GOOD.items() if k != "amount"}, {**GOOD, "amount": None},
                {**GOOD, "oldbalanceOrg": 5.0}, [], "x", None):
        assert code_of(assert_raises(AppError, validate_transaction, bad)) == "validation_error"


def test_invalid_numeric_values_rejected():
    for field, val in (("amount", -1), ("amount", float("nan")), ("amount", float("inf")), ("amount", "5"),
                       ("amount", True), ("amount", 1e13), ("step", -1), ("step", 1.5), ("step", "3"),
                       ("step", 10**9), ("step", True)):
        assert_raises(AppError, validate_transaction, {**GOOD, field: val})


def test_string_fields_reject_injection_and_length():
    for field, val in (("nameOrig", "x'; DROP TABLE transactions;--"), ("nameDest", "a b"), ("type", "=cmd|' /C calc'!A0"),
                       ("nameOrig", "A" * 65), ("type", ""), ("type", 5), ("nameDest", "../../etc/passwd")):
        assert_raises(AppError, validate_transaction, {**GOOD, field: val})


def test_validation_errors_do_not_echo_values():
    secret = "SECRET-VALUE-9f3"
    e = assert_raises(AppError, validate_transaction, {**GOOD, "nameOrig": secret + "!", "zzz" + secret: 1})
    assert secret not in json.dumps(e.body())


# ---------------- csv ----------------
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def small_csv(n=10, extra=True) -> bytes:
    df = make_synthetic(5, 5).head(n)
    return csv_bytes(df if extra else df[["step", "type", "amount", "nameOrig", "nameDest"]])


def test_csv_valid_ignores_extra_columns():
    df = parse_csv(small_csv(), 1_000_000, 100)
    assert list(df.columns) == ["step", "type", "amount", "nameOrig", "nameDest"] and len(df) == 10
    assert df["step"].dtype == np.int64 and df["amount"].dtype == np.float64


def test_csv_bom_and_whitespace_header_ok():
    raw = b"\xef\xbb\xbfstep, type ,amount,nameOrig,nameDest\n1,TRANSFER,5,C1,C2\n"
    assert len(parse_csv(raw, 1000, 10)) == 1


def test_csv_size_row_and_empty_limits():
    assert code_of(assert_raises(AppError, parse_csv, small_csv(), 50, 100)) == "file_too_large"
    assert code_of(assert_raises(AppError, parse_csv, small_csv(10), 1_000_000, 5)) == "too_many_rows"
    assert len(parse_csv(small_csv(5), 1_000_000, 5)) == 5  # exactly at the limit is allowed
    assert code_of(assert_raises(AppError, parse_csv, b"", 100, 5)) == "empty_file"
    assert code_of(assert_raises(AppError, parse_csv, b"step,type,amount,nameOrig,nameDest\n", 1000, 5)) == "no_rows"


def test_csv_schema_errors():
    e = assert_raises(AppError, parse_csv, b"step,type\n1,A\n", 1000, 5)
    assert code_of(e) == "missing_columns" and "amount" in e.details["missing"]
    assert code_of(assert_raises(AppError, parse_csv, b"step,step,type,amount,nameOrig,nameDest\n1,1,A,1,C1,C2\n",
                                 1000, 5)) == "duplicate_columns"


def test_csv_malformed_inputs():
    assert code_of(assert_raises(AppError, parse_csv, b"\xff\xfe\x00\x01", 1000, 5)) == "not_utf8_text"
    long_row = assert_raises(AppError, parse_csv, b"step,type,amount,nameOrig,nameDest\n1,A,1,C1,C2,EXTRA\n", 1000, 5)
    short_row = assert_raises(AppError, parse_csv, b"step,type,amount,nameOrig,nameDest\n1,A,1\n", 1000, 5)
    assert code_of(long_row) == code_of(short_row) == "malformed_csv" and short_row.details == {"row": 2}
    assert code_of(assert_raises(AppError, parse_csv, b'step,type,amount,nameOrig,nameDest\n1,"A,1,C1,C2\n', 1000, 5)) == "malformed_csv"
    assert code_of(assert_raises(AppError, parse_csv, b"a\x00b,c\n1,2\n", 1000, 5)) in ("not_csv", "malformed_csv")


def test_csv_invalid_rows_reported_by_row_number_without_values():
    raw = (b"step,type,amount,nameOrig,nameDest\n"
           b"1,TRANSFER,5,C1,C2\n"            # row 2 ok
           b"1,TRANSFER,nan,C1,C2\n"          # row 3 amount
           b"-4,TRANSFER,5,C1,C2\n"           # row 4 step
           b"1,TRANSFER,5,=EVIL(),C2\n"       # row 5 nameOrig (formula-like)
           b"1,TRANSFER,inf,C1,C2\n"          # row 6 amount
           b"1,,5,C1,C2\n")                   # row 7 empty type
    e = assert_raises(AppError, parse_csv, raw, 10_000, 50)
    assert code_of(e) == "invalid_rows" and e.details["total_invalid_rows"] == 5
    assert [d["row"] for d in e.details["first"]] == [3, 4, 5, 6, 7]
    assert "EVIL" not in json.dumps(e.body())


# ---------------- rate limiter ----------------
def test_rate_limiter_window_keys_and_retry_after():
    now = [0.0]
    rl = RateLimiter(2, 60.0, clock=lambda: now[0])
    assert rl.check("a")[0] and rl.check("a")[0]
    ok, retry = rl.check("a")
    assert not ok and 0 < retry <= 60
    assert rl.check("b")[0]  # other client unaffected
    now[0] = 61.0
    assert rl.check("a")[0]  # window expired


def test_rate_limiter_bounded_memory():
    rl = RateLimiter(1, 60.0, clock=lambda: 0.0, max_keys=50)
    for i in range(500):
        rl.check(f"k{i}")
    assert len(rl._hits) <= 51


# ---------------- sqlite repository ----------------
def row(i=0, flagged=False, type_="TRANSFER", score=0.5, step=1, orig=None):
    return dict(step=step, type=type_, amount=10.0 + i, nameOrig=orig or f"C{i}", nameDest="C9",
                score=score, threshold=0.7, flagged=flagged, type_known=True)


META = {"kind": "hist_gbm", "sha256": "abc", "data_label": "SYNTHETIC test fixture"}


def test_db_persistence_pagination_filters_sorting(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    rows = [row(i, flagged=(i % 4 == 0), score=i / 20, type_="TRANSFER" if i % 2 else "PAYMENT") for i in range(20)]
    ids = repo.insert_predictions(rows, "batch", "b1", META)
    assert ids == list(range(1, 21))
    p1 = repo.list_transactions(page=1, page_size=8)
    p3 = repo.list_transactions(page=3, page_size=8)
    assert p1["total"] == 20 and len(p1["items"]) == 8 and len(p3["items"]) == 4
    assert p1["items"][0]["id"] == 20  # newest first by default
    assert repo.list_transactions(page=9, page_size=8)["items"] == []
    assert repo.list_transactions(page=1, page_size=50, flagged=True)["total"] == 5
    assert repo.list_transactions(page=1, page_size=50, type_="PAYMENT")["total"] == 10
    assert repo.list_transactions(page=1, page_size=50, min_score=0.5, max_score=0.7)["total"] == 5
    assert repo.list_transactions(page=1, page_size=50, case_status="open")["total"] == 5
    top = repo.list_transactions(page=1, page_size=3, sort="score", order="desc")["items"]
    assert [t["score"] for t in top] == sorted([t["score"] for t in top], reverse=True)
    assert top[0]["model_data_label"] == "SYNTHETIC test fixture"


def test_db_sort_injection_rejected(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    for kw in ({"sort": "id; DROP TABLE transactions"}, {"order": "asc; --"}, {"sort": "name_orig"}):
        assert_raises(AppError, repo.list_transactions, page=1, page_size=5, **kw)
    assert repo.ping()


def test_db_case_updates_and_history(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    ids = repo.insert_predictions([row(0, flagged=True), row(1, flagged=False)], "single", None, META)
    assert repo.get_transaction(ids[0])["case_status"] == "open"
    assert repo.get_transaction(ids[1])["case_status"] is None
    repo.update_case(ids[0], "in_review", "checking")
    repo.update_case(ids[0], "confirmed_fraud", None)
    t = repo.get_transaction(ids[0])
    assert t["case_status"] == "confirmed_fraud"
    assert [(e["from_status"], e["to_status"]) for e in t["case_history"]] == [
        (None, "open"), ("open", "in_review"), ("in_review", "confirmed_fraud")]
    repo.update_case(ids[1], "open", "analyst opened")  # unflagged tx can get a case
    assert repo.get_transaction(ids[1])["case_status"] == "open"


def test_db_case_and_feedback_errors(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    (tid,) = repo.insert_predictions([row(0)], "single", None, META)
    assert code_of(assert_raises(AppError, repo.update_case, 999, "open", None)) == "not_found"
    assert code_of(assert_raises(AppError, repo.update_case, tid, "deleted", None)) == "validation_error"
    assert code_of(assert_raises(AppError, repo.set_feedback, 999, 1, None)) == "not_found"
    assert code_of(assert_raises(AppError, repo.set_feedback, tid, 2, None)) == "validation_error"
    assert code_of(assert_raises(AppError, repo.get_transaction, 999)) == "not_found"


def test_db_feedback_upsert_and_summary_counts(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    a, b = repo.insert_predictions([row(0, flagged=True, score=0.95), row(1, score=0.05)], "single", None, META)
    repo.set_feedback(a, 1, "looks bad")
    repo.set_feedback(a, 0, "actually fine")  # replaces, does not duplicate
    repo.set_feedback(b, 0, None)
    s = repo.summary()
    assert s["total_transactions"] == 2 and s["flagged_transactions"] == 1
    assert s["feedback"] == {"total": 2, "labelled_fraud": 0, "labelled_not_fraud": 2}
    hist = {h["bin_start"]: h["count"] for h in s["score_histogram"]}
    assert hist[0.9] == 1 and hist[0.0] == 1 and sum(hist.values()) == 2
    assert s["case_status_counts"]["open"] == 1


def test_db_rollback_leaves_no_partial_batch(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    bad = row(1)
    bad["flagged"] = 2  # violates CHECK (flagged IN (0,1)) on the 2nd row, after row 1 + its case were written
    assert_raises(sqlite3.IntegrityError, repo.insert_predictions, [row(0, flagged=True), bad], "batch", "b", META)
    assert repo.summary()["total_transactions"] == 0
    with repo._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 0


def test_db_numpy_values_are_accepted(tmp_path):
    repo = Repository(tmp_path / "a.db")
    repo.init_schema()
    r = row(0)
    r.update(step=np.int64(3), amount=np.float64(5.5), score=np.float32(0.25), flagged=np.bool_(False))
    (tid,) = repo.insert_predictions([r], "single", None, META)
    assert repo.get_transaction(tid)["step"] == 3


# ---------------- service: no model / invalid model ----------------
def test_no_model_blocks_prediction_without_fabricating(tmp_path):
    svc = make_service(tmp_path, None)
    e = assert_raises(AppError, svc.predict_one, GOOD)
    assert e.status == 503 and code_of(e) == "model_unavailable"
    assert code_of(assert_raises(AppError, svc.predict_batch, small_csv())) == "model_unavailable"
    assert svc.repo.summary()["total_transactions"] == 0  # nothing persisted


def test_invalid_input_is_reported_before_model_state(tmp_path):
    svc = make_service(tmp_path, None)
    assert code_of(assert_raises(AppError, svc.predict_one, {**GOOD, "amount": -1})) == "validation_error"
    assert code_of(assert_raises(AppError, svc.predict_batch, b"junk\n")) == "missing_columns"


def test_readiness_health_model_info_and_dashboard_empty_states(tmp_path):
    svc = make_service(tmp_path, None)
    assert svc.health() == {"status": "alive"}
    status, body = svc.readiness()
    assert status == 503 and body["ready"] is False
    assert body["checks"] == {"database": "ok", "model": "unavailable"}
    info = svc.model_info()
    assert info["state"] == "unavailable" and info["model"] is None
    d = svc.dashboard_summary()
    assert d["empty"] is True and d["total_transactions"] == 0 and d["flagged_transactions"] == 0
    assert all(h["count"] == 0 for h in d["score_histogram"]) and d["model"]["state"] == "unavailable"
    assert d["by_day"] == [] and d["feedback"]["total"] == 0
    assert svc.list_transactions()["items"] == [] and svc.alerts()["total"] == 0


def test_missing_and_empty_artifact_dirs_are_unavailable(tmp_path):
    assert make_service(tmp_path, tmp_path / "does-not-exist").model.state == "unavailable"
    (tmp_path / "empty").mkdir()
    assert make_service(tmp_path, tmp_path / "empty").model.state == "unavailable"


def copy_artifact(dst: Path) -> Path:
    import shutil
    shutil.copytree(trained_primary(), dst)
    return dst


def test_corrupt_tampered_and_incompatible_artifacts_are_invalid(tmp_path):
    d = copy_artifact(tmp_path / "m1")
    (d / "metadata.json").write_text("{nope")
    s = make_service(tmp_path / "s1" if (tmp_path / "s1").mkdir() is None else tmp_path, d)
    assert s.model.state == "invalid" and "corrupted" in s.model.reason
    d2 = copy_artifact(tmp_path / "m2")
    with (d2 / "model.joblib").open("ab") as f:
        f.write(b"x")
    (tmp_path / "s2").mkdir()
    s2 = make_service(tmp_path / "s2", d2)
    assert s2.model.state == "invalid" and "SHA-256" in s2.model.reason
    e = assert_raises(AppError, s2.predict_one, GOOD)
    assert e.status == 503 and e.details["model_state"] == "invalid"
    d3 = copy_artifact(tmp_path / "m3")
    meta = json.loads((d3 / "metadata.json").read_text())
    meta["library_versions"]["sklearn"] = "0.1.0"
    (d3 / "metadata.json").write_text(json.dumps(meta))
    (tmp_path / "s3").mkdir()
    s3 = make_service(tmp_path / "s3", d3)
    assert s3.model.state == "invalid" and "version mismatch" in s3.model.reason
    status, body = s3.readiness()
    assert status == 503 and body["checks"]["model"] == "invalid"


def test_error_reasons_do_not_leak_paths(tmp_path):
    d = copy_artifact(tmp_path / "m")
    (d / "metadata.json").write_text("{nope")
    s = make_service(tmp_path, d)
    blob = json.dumps(s.model.info()) + json.dumps(s.readiness()[1])
    assert str(tmp_path) not in blob and "Traceback" not in blob


# ---------------- service: ready model (artifact trained on SYNTHETIC data) ----------------
def test_predict_one_persists_and_labels_score_honestly(tmp_path):
    svc = ready_service(tmp_path)
    r = svc.predict_one(GOOD)
    assert 0.0 <= r["score"] <= 1.0 and r["flagged"] == (r["score"] >= r["threshold"])
    assert "not a probability" in r["score_note"] and "probability" not in {k for k in r}
    assert r["model"]["metrics_origin"] == "SYNTHETIC test fixture"
    t = svc.get_transaction(r["transaction_id"])
    assert t["amount"] == 1234.5 and t["model_data_label"] == "SYNTHETIC test fixture"
    assert svc.dashboard_summary()["total_transactions"] == 1


def test_model_info_exposes_provenance_of_metrics(tmp_path):
    info = ready_service(tmp_path).model_info()
    assert info["state"] == "ready" and info["model"]["metrics"]["metrics_are_synthetic"] is True
    assert info["model"]["metrics"]["origin"] == "SYNTHETIC test fixture"
    assert "test" in info["model"]["metrics"] and "not a calibrated probability" in info["model"]["score_semantics"].lower()


def test_readiness_ok_with_model(tmp_path):
    status, body = ready_service(tmp_path).readiness()
    assert status == 200 and body["ready"] is True


def test_unknown_type_warns(tmp_path):
    r = ready_service(tmp_path).predict_one({**GOOD, "type": "BRAND_NEW"})
    assert r["warnings"] and "unknown" in r["warnings"][0]


def test_single_and_batch_use_the_same_model_path(tmp_path):
    svc = ready_service(tmp_path)
    raw = small_csv(12)
    out = svc.predict_batch(raw)
    assert out["rows"] == 12 and len(out["results"]) == 12 and out["results"][0]["row"] == 2
    df = parse_csv(raw, 10**6, 100)
    direct = svc.model.loaded.predict_score(df)
    assert np.allclose([r["score"] for r in out["results"]], direct)
    assert svc.repo.summary()["total_transactions"] == 12
    one = ready_service(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path).predict_one(
        {k: df.iloc[0][k].item() if hasattr(df.iloc[0][k], "item") else df.iloc[0][k] for k in df.columns})
    assert np.isclose(one["score"], direct[0])


def test_batch_ignores_balance_columns(tmp_path):
    svc = ready_service(tmp_path)
    base = make_synthetic(5, 5).head(10)
    a = svc.predict_batch(csv_bytes(base))
    altered = base.assign(newbalanceOrig=1e9, oldbalanceDest=-5, isFlaggedFraud=1, oldbalanceOrg=0, newbalanceDest=7)
    (tmp_path / "y").mkdir()
    b = ready_service(tmp_path / "y").predict_batch(csv_bytes(altered))
    assert [r["score"] for r in a["results"]] == [r["score"] for r in b["results"]]


def test_history_from_earlier_scored_transactions_is_used(tmp_path):
    svc = ready_service(tmp_path)
    svc.predict_one({**GOOD, "step": 3, "amount": 100.0, "nameOrig": "HIST1"})
    r = svc.predict_one({**GOOD, "step": 9, "amount": 500.0, "nameOrig": "HIST1"})
    lm = svc.model.loaded
    frame = pd.DataFrame([dict(step=3, type="TRANSFER", amount=100.0, nameOrig="HIST1", nameDest="C2"),
                          dict(step=9, type="TRANSFER", amount=500.0, nameOrig="HIST1", nameDest="C2")])
    assert np.isclose(r["score"], lm.predict_score(frame)[-1])


def test_alerts_cases_and_feedback_flow(tmp_path):
    svc = ready_service(tmp_path)
    out = svc.predict_batch(csv_bytes(make_synthetic(10, 20)))
    flagged_ids = [r["transaction_id"] for r in out["results"] if r["flagged"]]
    assert out["flagged"] == len(flagged_ids)
    if not flagged_ids:  # synthetic data is easy; guard against a degenerate threshold
        return
    al = svc.alerts(page_size=5)
    assert al["total"] == len(flagged_ids) and all(a["case_status"] == "open" for a in al["items"])
    tid = flagged_ids[0]
    assert svc.update_case(tid, {"status": "in_review", "note": "  look  "})["note"] == "look"
    assert svc.alerts(case_status="in_review")["total"] == 1
    assert svc.set_feedback(tid, {"label": 1})["label"] == 1
    assert svc.get_transaction(tid)["feedback_label"] == 1
    assert svc.dashboard_summary()["feedback"]["labelled_fraud"] == 1


def test_service_input_validation_for_cases_feedback_and_paging(tmp_path):
    svc = make_service(tmp_path, None)
    (tid,) = svc.repo.insert_predictions([row(0)], "single", None, META)
    for fn, args in ((svc.update_case, (tid, {"status": "nope"})), (svc.update_case, (tid, {"note": "x"})),
                     (svc.update_case, (tid, {"status": "open", "extra": 1})),
                     (svc.update_case, (tid, {"status": "open", "note": "x" * 501})),
                     (svc.update_case, (tid, "open")), (svc.set_feedback, (tid, {"label": 2})),
                     (svc.set_feedback, (tid, {"label": True})), (svc.set_feedback, (tid, {})),
                     (svc.get_transaction, (0,)), (svc.get_transaction, ("1",)),
                     (svc.update_case, (True, {"status": "open"}))):
        assert code_of(assert_raises(AppError, fn, *args)) == "validation_error", (fn.__name__, args)
    for kw in ({"page": 0}, {"page_size": 0}, {"page_size": 10_000}, {"page": "1"}, {"source": "x"},
               {"case_status": "x"}, {"sort": "bogus"}, {"bogus": 1}):
        assert code_of(assert_raises(AppError, svc.list_transactions, **kw)) == "validation_error", kw


def test_logs_contain_no_transaction_data(tmp_path):
    records: list[str] = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger("fraudapp")
    h = H()
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        svc = ready_service(tmp_path)
        svc.predict_one({**GOOD, "nameOrig": "SECRETNAME123", "amount": 987654.321})
        svc.predict_batch(csv_bytes(make_synthetic(3, 5)))
        try:
            svc.predict_one({**GOOD, "nameDest": "SECRETDEST!!"})
        except AppError:
            pass
    finally:
        lg.removeHandler(h)
    blob = "\n".join(records)
    assert records and "SECRETNAME123" not in blob and "987654" not in blob and "SECRETDEST" not in blob
