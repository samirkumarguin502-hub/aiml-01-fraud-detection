"""Streaming tests. Real HTTP against a local stdlib FAKE API server; fake Kafka client objects.
NOT tested: a real Kafka broker or the confluent-kafka library. Data is SYNTHETIC."""
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

from streaming.__main__ import main  # noqa: E402
from streaming.messages import MessageError, decode_message, encode_message  # noqa: E402
from streaming.outcomes import OutcomeRecorder  # noqa: E402
from streaming.replay import ApiSink, KafkaSink, consume_loop, load_rows, replay  # noqa: E402

TX = dict(step=3, type="TRANSFER", amount=50.0, nameOrig="SECRETORIG1", nameDest="SECRETDEST1")


# ---- fake API server ----
class FakeApi:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                outer.calls.append((self.path, self.headers.get("Content-Type"), body))
                status, payload, headers = (outer.script.pop(0) if outer.script else (200, {"flagged": False}, {}))
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/api/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def dead_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/api/v1"


# ---- messages ----
def test_message_roundtrip():
    seq, tx = decode_message(encode_message(TX, 7))
    assert seq == 7 and tx == {**TX, "step": 3, "amount": 50.0}


def test_malformed_messages_are_rejected_without_leaking_content():
    good = json.loads(encode_message(TX, 1))
    cases = {"not_bytes": "str", "bad_size": b"", "not_json": b"{nope", "bad_schema": b'{"schema":99}',
             "bad_seq": json.dumps({**good, "seq": "x"}).encode(),
             "invalid_transaction": json.dumps({**good, "tx": {**good["tx"], "amount": -1}}).encode()}
    for code, raw in cases.items():
        e = assert_raises(MessageError, decode_message, raw)
        assert e.code == code and "SECRET" not in str(e), code
    leaky = json.dumps({**good, "tx": {**good["tx"], "newbalanceOrig": 0}}).encode()  # extra field rejected
    assert assert_raises(MessageError, decode_message, leaky).code == "invalid_transaction"
    assert assert_raises(MessageError, decode_message, b"x" * 5000).code == "bad_size"


# ---- loading rows ----
def test_load_rows_sorted_and_validated(tmp_path):
    f = tmp_path / "s.csv"
    make_synthetic(5, 4).sample(frac=1, random_state=1).to_csv(f, index=False)
    rows = load_rows(f)
    assert [r["step"] for r in rows] == sorted(r["step"] for r in rows) and set(rows[0]) == {
        "step", "type", "amount", "nameOrig", "nameDest"}
    from ml.train import DatasetError
    assert_raises(DatasetError, load_rows, tmp_path / "missing.csv")


# ---- pacing ----
class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def rows_at(*steps):
    return [{**TX, "step": s} for s in steps]


def test_replay_pacing_follows_step_gaps_and_speed():
    c, out = Clock(), []
    replay(rows_at(1, 1, 2, 5), lambda r, i: out.append((r["step"], i)), speed=3600.0, sleep=c.sleep, clock=c.now)
    assert out == [(1, 0), (1, 1), (2, 2), (5, 3)]
    assert abs(sum(c.sleeps) - 4.0) < 1e-6  # gaps 1 + 3 steps at 3600 sim-sec per real sec = 4 s


def test_replay_speed_zero_never_sleeps_and_limit_and_stop():
    c = Clock()
    assert replay(rows_at(1, 5, 9), lambda r, i: None, speed=0, sleep=c.sleep, clock=c.now) == 3 and c.sleeps == []
    assert replay(rows_at(1, 2, 3), lambda r, i: None, limit=2) == 2
    stop, seen = threading.Event(), []

    def emit(r, i):
        seen.append(i)
        if i == 1:
            stop.set()
    replay(rows_at(1, 2, 3, 4), emit, stop=stop)
    assert seen == [0, 1]


# ---- API sink (real HTTP to a fake API) ----
def test_api_sink_outcomes_and_no_content_in_logs(tmp_path):
    api = FakeApi([(200, {"flagged": True}, {}), (200, {"flagged": False}, {}),
                   (422, {"error": {"code": "validation_error"}}, {}), (503, {"error": {"code": "model_unavailable"}}, {})])
    try:
        rec = OutcomeRecorder(tmp_path / "o", "api")
        sink = ApiSink(api.url, rec)
        for i in range(4):
            sink.send_one(TX, i)
        s = rec.close()
    finally:
        api.close()
    assert s["outcomes"] == {"scored_flagged": 1, "scored_ok": 1, "api_422_validation_error": 1,
                             "api_503_model_unavailable": 1}
    assert api.calls[0][0] == "/api/v1/predict" and api.calls[0][1] == "application/json"
    logs = (tmp_path / "o" / "outcomes.jsonl").read_text() + (tmp_path / "o" / "last_run_summary.json").read_text()
    assert "SECRET" not in logs and "50.0" not in logs


def test_api_sink_retries_on_429_using_retry_after():
    api = FakeApi([(429, {"error": {"code": "rate_limited"}}, {"Retry-After": "7"}), (200, {"flagged": False}, {})])
    sleeps = []
    try:
        rec = OutcomeRecorder(None, "api")
        ApiSink(api.url, rec, sleep=sleeps.append).send_one(TX, 0)
    finally:
        api.close()
    assert sleeps == [7.0] and rec.counts == {"scored_ok": 1} and len(api.calls) == 2


def test_api_sink_gives_up_after_max_retries_and_handles_unreachable():
    api = FakeApi([(429, {"error": {"code": "rate_limited"}}, {"Retry-After": "1"})] * 10)
    try:
        rec = OutcomeRecorder(None, "api")
        ApiSink(api.url, rec, max_retries=2, sleep=lambda s: None).send_one(TX, 0)
    finally:
        api.close()
    assert rec.counts == {"api_429_rate_limited": 1} and len(api.calls) == 3
    rec2 = OutcomeRecorder(None, "api")
    ApiSink(dead_url(), rec2, timeout=1).send_one(TX, 0)
    assert rec2.counts == {"api_unreachable": 1}


def test_api_sink_batch_sends_csv_and_counts_each_row():
    api = FakeApi([(200, {"results": [{"flagged": True}, {"flagged": False}, {"flagged": False}]}, {})])
    try:
        rec = OutcomeRecorder(None, "api")
        ApiSink(api.url, rec).send_batch(rows_at(1, 2, 3), 0)
    finally:
        api.close()
    path, ctype, body = api.calls[0]
    assert path.endswith("/predict/batch") and ctype == "text/csv" and body.splitlines()[0] == b"step,type,amount,nameOrig,nameDest"
    assert rec.counts == {"scored_flagged": 1, "scored_ok": 2}


def test_api_sink_rejects_remote_plain_http():
    assert_raises(ValueError, ApiSink, "http://example.com/api", OutcomeRecorder(None, "x"))


# ---- Kafka (FAKE clients: verifies our logic only, not Kafka) ----
class FakeProducer:
    def __init__(self):
        self.sent, self.cb = [], []

    def produce(self, topic, value=None, key=None, callback=None):
        self.sent.append((topic, value, key))
        self.cb.append(callback)

    def poll(self, t):
        for cb in self.cb:
            cb(None, None)
        self.cb.clear()

    def flush(self, t):
        self.poll(0)


def test_kafka_sink_publishes_valid_and_skips_invalid():
    rec, p = OutcomeRecorder(None, "produce"), FakeProducer()
    ks = KafkaSink("x:9092", "tx", rec, producer=p)
    ks.send_one(TX, 0)
    ks.send_one({**TX, "amount": -5}, 1)
    ks.flush()
    assert len(p.sent) == 1 and p.sent[0][0] == "tx" and p.sent[0][2] == b"SECRETORIG1"
    assert rec.counts == {"published": 1, "invalid_row_skipped": 1}
    assert decode_message(p.sent[0][1])[1]["amount"] == 50.0


def test_kafka_sink_without_client_library_explains():
    try:
        import confluent_kafka  # noqa: F401
        return  # library present: nothing to assert here
    except ImportError:
        pass
    e = assert_raises(RuntimeError, KafkaSink, "x", "t", OutcomeRecorder(None, "x"))
    assert "--mode api" in str(e) or "mode api" in str(e)


class Msg:
    def __init__(self, value=None, err=None):
        self._v, self._e = value, err

    def value(self):
        return self._v

    def error(self):
        return self._e


class FakeConsumer:
    def __init__(self, msgs):
        self.msgs, self.committed = list(msgs), []

    def poll(self, t):
        return self.msgs.pop(0) if self.msgs else None

    def commit(self, msg, asynchronous=False):
        self.committed.append(msg)


def test_consume_loop_validates_commits_after_processing_and_survives_bad_messages():
    api = FakeApi([(200, {"flagged": True}, {}), (200, {"flagged": False}, {})])
    good1, good2 = Msg(encode_message(TX, 1)), Msg(encode_message({**TX, "step": 4}, 2))
    bad, kerr = Msg(b"{broken"), Msg(None, err="boom")
    cons = FakeConsumer([good1, bad, kerr, good2])
    try:
        rec = OutcomeRecorder(None, "consume")
        n = consume_loop(cons, ApiSink(api.url, rec), rec, threading.Event(), max_messages=3)
    finally:
        api.close()
    assert n == 3 and cons.committed == [good1, bad, good2]  # error message not committed
    assert rec.counts == {"scored_flagged": 1, "malformed_not_json": 1, "kafka_error": 1, "scored_ok": 1}
    assert len(api.calls) == 2  # the malformed message never reached the API


def test_consume_loop_stops_gracefully_on_event():
    stop = threading.Event()
    cons = FakeConsumer([])
    t = threading.Timer(0.2, stop.set)
    t.start()
    n = consume_loop(cons, ApiSink("http://127.0.0.1:1/x", OutcomeRecorder(None, "c")), OutcomeRecorder(None, "c"),
                     stop, poll_timeout=0.01)
    t.cancel()
    assert n == 0 and stop.is_set()


# ---- CLI ----
def test_cli_api_mode_end_to_end_against_fake_api(tmp_path):
    f = tmp_path / "s.csv"
    make_synthetic(3, 4).to_csv(f, index=False)
    api = FakeApi([])
    try:
        code = main(["api", "--data", str(f), "--api-url", api.url, "--outcomes-dir", str(tmp_path / "o"),
                     "--limit", "5"])
    finally:
        api.close()
    s = json.loads((tmp_path / "o" / "last_run_summary.json").read_text())
    assert code == 0 and s["processed"] == 5 and s["outcomes"] == {"scored_ok": 5} and len(api.calls) == 5


def test_cli_batch_mode_and_error_paths(tmp_path):
    f = tmp_path / "s.csv"
    make_synthetic(3, 4).to_csv(f, index=False)
    api = FakeApi([(200, {"results": [{"flagged": False}] * 5}, {}), (200, {"results": [{"flagged": False}] * 5}, {}),
                   (200, {"results": [{"flagged": False}] * 2}, {})])
    try:
        code = main(["api", "--data", str(f), "--api-mode", "batch", "--batch-size", "5", "--api-url", api.url,
                     "--outcomes-dir", str(tmp_path / "o2")])
    finally:
        api.close()
    assert code == 0 and [c[0].endswith("batch") for c in api.calls] == [True] * 3
    assert main(["api", "--outcomes-dir", str(tmp_path / "o3")]) == 2  # --data missing
    assert main(["api", "--data", str(tmp_path / "nope.csv"), "--outcomes-dir", str(tmp_path / "o4")]) == 2
    try:
        import confluent_kafka  # noqa: F401
    except ImportError:
        assert main(["consume", "--outcomes-dir", str(tmp_path / "o5")]) == 2
        assert main(["produce", "--data", str(f), "--outcomes-dir", str(tmp_path / "o6")]) == 2
