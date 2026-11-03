"""Replay real dataset rows (from a validated local CSV) at a configurable speed.

Timing: PaySim `step` = 1 simulated hour (3600 s). `speed` = simulated seconds per real second, so the
real delay between two consecutive distinct steps is (delta_steps * 3600) / speed. speed=0 means no
delay (as fast as possible). Rows sharing a step are emitted back to back.
Sinks: ApiSink (HTTP to the FastAPI service, stdlib only) and KafkaSink (optional confluent-kafka).
"""
from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Iterable, Iterator

from backend.app.validation import ALL_FIELDS, TX_FIELDS
from ml.pipeline.schema import feature_set_for_fields
from streaming.messages import MessageError, decode_message, encode_message
from streaming.outcomes import OutcomeRecorder

log = logging.getLogger("fraudapp.streaming")


def load_rows(csv_path, nrows: int | None = None, fields: tuple = TX_FIELDS) -> list[dict]:
    """Rows from a CSV via the SAME loader/validation as training. Sorted by step (stable).
    `fields` = what the target API model needs; only those columns are loaded (see ApiSink.discover_fields)."""
    from ml.train import load_dataset  # raises DatasetError / SchemaError with clear messages
    df = load_dataset(csv_path, nrows, feature_set_for_fields(fields))
    df = df.sort_values("step", kind="stable")
    return df[list(fields)].to_dict("records")


def replay(rows: Iterable[dict], emit: Callable[[dict, int], None], *, speed: float = 0.0,
           stop: threading.Event | None = None, sleep: Callable[[float], None] = time.sleep,
           limit: int | None = None, clock: Callable[[], float] = time.monotonic) -> int:
    """Call emit(row, seq) for each row, pacing by step gaps. Returns rows emitted."""
    stop = stop or threading.Event()
    n, prev_step = 0, None
    for seq, row in enumerate(rows):
        if stop.is_set() or (limit is not None and n >= limit):
            break
        step = int(row["step"])
        if speed > 0 and prev_step is not None and step > prev_step:
            wait = (step - prev_step) * 3600.0 / speed
            end = clock() + wait
            while not stop.is_set() and clock() < end:  # interruptible wait
                sleep(min(0.2, max(0.0, end - clock())))
            if stop.is_set():
                break
        prev_step = step
        emit(row, seq)
        n += 1
    return n


class ApiSink:
    """Sends transactions to the API. Records only outcomes, never content."""

    def __init__(self, base_url: str, recorder: OutcomeRecorder, *, timeout: float = 10.0,
                 max_retries: int = 5, sleep: Callable[[float], None] = time.sleep,
                 stop: threading.Event | None = None):
        if not base_url.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            raise ValueError("base_url must be https:// or a loopback http:// URL")
        self.base, self.rec, self.timeout = base_url.rstrip("/"), recorder, timeout
        self.max_retries, self.sleep, self.stop = max_retries, sleep, stop or threading.Event()
        self.fields: tuple = TX_FIELDS

    def discover_fields(self) -> tuple:
        """Ask the API which inputs its loaded model needs (GET /model -> model.input_fields).
        Falls back to the base five fields when the API is unreachable, has no model, or answers oddly."""
        req = urllib.request.Request(self.base + "/model", method="GET", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - scheme checked in __init__
                data = json.loads(r.read(1_000_000) or b"{}")
            fields = (data.get("model") or {}).get("input_fields")
            if isinstance(fields, list) and fields and all(f in ALL_FIELDS for f in fields) and set(TX_FIELDS) <= set(fields):
                self.fields = tuple(f for f in ALL_FIELDS if f in fields)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, AttributeError):
            pass
        return self.fields

    def _post(self, path: str, body: bytes, ctype: str) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=body, method="POST", headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - scheme checked above
                return r.status, json.loads(r.read(5_000_000) or b"{}")
        except urllib.error.HTTPError as e:
            try:
                data = json.loads(e.read(100_000) or b"{}")
            except ValueError:
                data = {}
            e.retry_after = e.headers.get("Retry-After") if e.headers else None
            return e.code, {**data, "_retry_after": e.retry_after}
        except (urllib.error.URLError, TimeoutError, OSError):
            return 0, {}

    def _with_retry(self, path: str, body: bytes, ctype: str) -> tuple[int, dict]:
        for attempt in range(self.max_retries + 1):
            status, data = self._post(path, body, ctype)
            if status != 429 or attempt == self.max_retries or self.stop.is_set():
                return status, data
            try:
                wait = min(float(data.get("_retry_after") or 1), 30.0)
            except ValueError:
                wait = 1.0
            log.warning("rate limited; retrying in %.0fs", wait)
            self.sleep(wait)
        return status, data

    def _classify(self, status: int, data: dict) -> str:
        if status == 200:
            return "scored_flagged" if data.get("flagged") else "scored_ok"
        if status == 0:
            return "api_unreachable"
        code = (data.get("error") or {}).get("code") if isinstance(data.get("error"), dict) else None
        return f"api_{status}_{code or 'error'}"

    def send_one(self, row: dict, seq: int) -> None:
        body = json.dumps({k: row[k] for k in self.fields}).encode()
        status, data = self._with_retry("/predict", body, "application/json")
        self.rec.record(self._classify(status, data), seq, status)

    def send_batch(self, rows: list[dict], first_seq: int) -> None:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.fields)
        for r in rows:
            w.writerow([r[k] for k in self.fields])
        status, data = self._with_retry("/predict/batch", buf.getvalue().encode(), "text/csv")
        if status == 200:
            for i, res in enumerate(data.get("results", [])):
                self.rec.record("scored_flagged" if res.get("flagged") else "scored_ok", first_seq + i, 200)
        else:
            for i in range(len(rows)):
                self.rec.record(self._classify(status, data), first_seq + i, status)


class KafkaSink:
    """Publishes validated messages. Requires confluent-kafka; NOT verified against a real broker."""

    def __init__(self, bootstrap: str, topic: str, recorder: OutcomeRecorder, producer=None, fields: tuple = TX_FIELDS):
        self.topic, self.rec, self.fields = topic, recorder, fields
        if producer is None:
            try:
                from confluent_kafka import Producer
            except ImportError as e:
                raise RuntimeError("confluent-kafka is not installed (python -m pip install confluent-kafka). "
                                   "Use --mode api for the Kafka-free demo.") from e
            producer = Producer({"bootstrap.servers": bootstrap, "client.id": "aiml01-replay"})
        self.producer = producer

    def _on_delivery(self, err, msg) -> None:
        self.rec.record("kafka_delivery_failed" if err is not None else "published")

    def send_one(self, row: dict, seq: int) -> None:
        try:
            value = encode_message(row, seq, self.fields)
        except Exception:  # noqa: BLE001 - validation failure: skip, count, never log content
            self.rec.record("invalid_row_skipped", seq)
            return
        self.producer.produce(self.topic, value=value, key=str(row["nameOrig"]).encode(),
                              callback=self._on_delivery)
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush(10)


def consume_loop(consumer, sink: ApiSink, rec: OutcomeRecorder, stop: threading.Event,
                 poll_timeout: float = 1.0, max_messages: int | None = None, fields: tuple | None = None) -> int:
    """Poll -> validate -> score via API -> commit AFTER processing (at-least-once). Malformed
    messages are counted and committed (skipped) so they cannot block the partition."""
    handled = 0
    while not stop.is_set() and (max_messages is None or handled < max_messages):
        msg = consumer.poll(poll_timeout)
        if msg is None:
            continue
        err = msg.error()
        if err:
            rec.record("kafka_error")
            log.error("kafka error received")
            continue
        try:
            seq, tx = decode_message(msg.value(), fields or sink.fields)
        except MessageError as e:
            rec.record(f"malformed_{e.code}")
            consumer.commit(msg, asynchronous=False)
            handled += 1
            continue
        sink.send_one(tx, seq if seq is not None else handled)
        consumer.commit(msg, asynchronous=False)
        handled += 1
    return handled
