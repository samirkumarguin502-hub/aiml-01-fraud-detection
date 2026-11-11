"""HTTP-level tests for the FastAPI binding (backend/app/main.py). SYNTHETIC data only.

These need fastapi + httpx (see requirements-api.txt / requirements-dev.txt). In an environment
without them every test is SKIPPED, which means the FastAPI layer is UNVERIFIED there.
Written without being executed in the authoring sandbox; expect to fix small issues on first run.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_synthetic  # noqa: E402

try:
    from fastapi.testclient import TestClient  # needs fastapi + httpx
    HAVE_HTTP = True
except Exception:  # noqa: BLE001
    HAVE_HTTP = False

GOOD = dict(step=5, type="TRANSFER", amount=1234.5, nameOrig="C1", nameDest="C2")


def client(tmp_path, with_model=False, **over):
    if not HAVE_HTTP:
        raise unittest.SkipTest("BLOCKED: fastapi/httpx not installed; FastAPI layer unverified")
    from backend.app.config import Settings
    from backend.app.main import create_app
    from test_backend_core import trained_primary  # same module name as other tests: one cached artifact
    s = Settings(database_path=tmp_path / "h.db", model_dir=trained_primary() if with_model else None,
                 app_env="test", **over)
    return TestClient(create_app(s))


def err(resp):
    return resp.json()["error"]


def test_health_ready_and_headers(tmp_path):
    c = client(tmp_path)
    r = c.get("/api/v1/health")
    assert r.status_code == 200 and r.json() == {"status": "alive"}
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-request-id"]
    r = c.get("/api/v1/ready")
    assert r.status_code == 503 and r.json()["checks"]["model"] == "unavailable"


def test_ready_200_with_model(tmp_path):
    assert client(tmp_path, True).get("/api/v1/ready").status_code == 200


def test_predict_without_model_is_503_not_a_fake_score(tmp_path):
    r = client(tmp_path).post("/api/v1/predict", json=GOOD)
    assert r.status_code == 503 and err(r)["code"] == "model_unavailable" and "score" not in r.json()


def test_predict_with_model_and_persistence(tmp_path):
    c = client(tmp_path, True)
    r = c.post("/api/v1/predict", json=GOOD)
    assert r.status_code == 200 and 0 <= r.json()["score"] <= 1 and "not a probability" in r.json()["score_note"]
    t = c.get(f"/api/v1/transactions/{r.json()['transaction_id']}")
    assert t.status_code == 200 and t.json()["amount"] == 1234.5
    assert c.get("/api/v1/dashboard/summary").json()["total_transactions"] == 1


def test_invalid_requests_unified_error_without_echo(tmp_path):
    c = client(tmp_path, True)
    secret = "SECRET-ECHO-CHECK"
    for body in ({**GOOD, "amount": -5}, {**GOOD, "nameOrig": secret + "'; DROP"}, {"step": 1},
                 {**GOOD, "oldbalanceOrg": 1}, {**GOOD, "amount": "abc"}):
        r = c.post("/api/v1/predict", json=body)
        assert r.status_code == 422 and err(r)["code"] == "validation_error"
        assert secret not in r.text and "Traceback" not in r.text
    r = c.post("/api/v1/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert c.get("/api/v1/nope").status_code == 404 and err(c.get("/api/v1/nope"))["code"] == "not_found"
    assert c.delete("/api/v1/health").status_code == 405


def test_json_body_limit(tmp_path):
    r = client(tmp_path, True, max_json_bytes=300).post("/api/v1/predict", json={**GOOD, "type": "A" * 1000})
    assert r.status_code == 413 and err(r)["code"] == "payload_too_large"


def csv_body(n=10):
    return make_synthetic(5, 5).head(n).to_csv(index=False).encode()


def test_batch_upload_flow_and_limits(tmp_path):
    c = client(tmp_path, True, max_upload_bytes=100_000, max_csv_rows=15, rate_batch_per_min = 20,)
    h = {"content-type": "text/csv"}
    ok = c.post("/api/v1/predict/batch", content=csv_body(10), headers=h)
    assert ok.status_code == 200 and ok.json()["rows"] == 10
    assert c.post("/api/v1/predict/batch", content=csv_body(20), headers=h).status_code == 413
    assert c.post("/api/v1/predict/batch", content=b"x" * 200_000, headers=h).status_code == 413
    assert c.post("/api/v1/predict/batch", content=b"a,b\n1,2\n", headers=h).status_code == 422
    assert c.post("/api/v1/predict/batch", content=b"\xff\xfe\x00", headers=h).status_code == 400
    assert c.post("/api/v1/predict/batch", content=csv_body(5), headers={"content-type": "image/png"}).status_code == 415
    assert c.post("/api/v1/predict/batch", content=b"", headers=h).status_code == 400


def test_batch_without_model_503(tmp_path):
    r = client(tmp_path).post("/api/v1/predict/batch", content=csv_body(3), headers={"content-type": "text/csv"})
    assert r.status_code == 503 and err(r)["code"] == "model_unavailable"


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    c = client(tmp_path, False, rate_predict_per_min=3)
    codes = [c.post("/api/v1/predict", json=GOOD).status_code for _ in range(5)]
    assert codes[:3] == [503, 503, 503] and codes[3:] == [429, 429]
    r = c.post("/api/v1/predict", json=GOOD)
    assert "retry-after" in r.headers and err(r)["code"] == "rate_limited"


def test_cors_allows_only_configured_origins(tmp_path):
    c = client(tmp_path, cors_origins=("http://good.example",))
    ok = c.get("/api/v1/health", headers={"Origin": "http://good.example"})
    bad = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://good.example"
    assert "access-control-allow-origin" not in bad.headers


def test_optional_api_key_is_required_only_when_configured(tmp_path):
    c = client(tmp_path, True, api_key="secret-key")
    assert c.get("/api/v1/health").status_code == 200
    assert c.get("/api/v1/ready").status_code == 200
    denied = c.post("/api/v1/predict", json=GOOD)
    assert denied.status_code == 401 and err(denied)["code"] == "unauthorized"
    assert denied.headers["www-authenticate"] == "Bearer"
    allowed = c.post("/api/v1/predict", json=GOOD, headers={"X-API-Key": "secret-key"})
    assert allowed.status_code == 200


def test_cases_feedback_alerts_and_pagination_over_http(tmp_path):
    c = client(tmp_path, True)
    c.post("/api/v1/predict/batch", content=make_synthetic(10, 20).to_csv(index=False).encode(),
           headers={"content-type": "text/csv"})
    page = c.get("/api/v1/transactions", params={"page": 2, "page_size": 50}).json()
    assert page["total"] == 200 and len(page["items"]) == 50 and page["page"] == 2
    assert c.get("/api/v1/transactions", params={"page_size": 100000}).status_code == 422
    assert c.get("/api/v1/transactions", params={"sort": "id; DROP TABLE x"}).status_code == 422
    r = c.patch("/api/v1/transactions/1/case", json={"status": "in_review", "note": "n"})
    assert r.status_code == 200 and r.json()["status"] == "in_review"
    assert c.patch("/api/v1/transactions/1/case", json={"status": "bogus"}).status_code == 422
    assert c.patch("/api/v1/transactions/99999/case", json={"status": "open"}).status_code == 404
    assert c.put("/api/v1/transactions/1/feedback", json={"label": 1}).status_code == 200
    assert c.put("/api/v1/transactions/1/feedback", json={"label": 5}).status_code == 422
    assert c.get("/api/v1/alerts").status_code == 200


def test_model_and_empty_dashboard_without_model(tmp_path):
    c = client(tmp_path)
    m = c.get("/api/v1/model").json()
    assert m["state"] == "unavailable" and m["model"] is None
    d = c.get("/api/v1/dashboard/summary").json()
    assert d["empty"] is True and d["total_transactions"] == 0
    assert c.get("/api/v1/alerts").json()["items"] == []


def test_docs_disabled_in_production(tmp_path):
    if not HAVE_HTTP:
        raise unittest.SkipTest("BLOCKED: fastapi/httpx not installed; FastAPI layer unverified")
    from backend.app.config import Settings
    from backend.app.main import create_app
    app = create_app(Settings(database_path=tmp_path / "p.db", app_env="production"))
    assert TestClient(app).get("/docs").status_code == 404
