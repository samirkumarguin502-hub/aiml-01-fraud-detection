"""Replay validated dataset rows to FastAPI or Kafka."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Iterable

from backend.app.validation import ALL_FIELDS, TX_FIELDS
from ml.pipeline.schema import feature_set_for_fields
from streaming.messages import MessageError, decode_message, encode_message
from streaming.outcomes import OutcomeRecorder

log = logging.getLogger("fraudapp.streaming")


def load_rows(
    csv_path,
    nrows: int | None = None,
    fields: tuple = TX_FIELDS,
) -> list[dict]:
    """Load validated CSV rows using the training dataset loader."""
    from ml.train import load_dataset

    df = load_dataset(
        csv_path,
        nrows,
        feature_set_for_fields(fields),
    )
    df = df.sort_values("step", kind="stable")
    return df[list(fields)].to_dict("records")


def replay(
    rows: Iterable[dict],
    emit: Callable[[dict, int], None],
    *,
    speed: float = 0.0,
    stop: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limit: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Emit rows with optional simulated-time pacing."""
    stop = stop or threading.Event()
    n = 0
    prev_step = None

    for seq, row in enumerate(rows):
        if stop.is_set() or (limit is not None and n >= limit):
            break

        step = int(row["step"])

        if speed > 0 and prev_step is not None and step > prev_step:
            wait = (step - prev_step) * 3600.0 / speed
            end = clock() + wait

            while not stop.is_set() and clock() < end:
                sleep(min(0.2, max(0.0, end - clock())))

            if stop.is_set():
                break

        prev_step = step
        emit(row, seq)
        n += 1

    return n


class ApiSink:
    """Sends authenticated transactions to FastAPI."""

    def __init__(
        self,
        base_url: str,
        recorder: OutcomeRecorder,
        *,
        timeout: float = 10.0,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        stop: threading.Event | None = None,
        api_key: str | None = None,
    ):
        if not base_url.startswith(
            ("http://127.0.0.1", "http://localhost", "https://")
        ):
            raise ValueError(
                "base_url must be https:// or a loopback http:// URL"
            )

        self.base = base_url.rstrip("/")
        self.rec = recorder
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep = sleep
        self.stop = stop or threading.Event()

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("API_KEY", "")
        ).strip()

        self.fields: tuple = TX_FIELDS

    def _headers(
        self,
        content_type: str | None = None,
    ) -> dict:
        headers = {"Accept": "application/json"}

        if content_type:
            headers["Content-Type"] = content_type

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    def discover_fields(self) -> tuple:
        """Discover the loaded model's input fields."""
        req = urllib.request.Request(
            self.base + "/model",
            method="GET",
            headers=self._headers(),
        )

        try:
            with urllib.request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                data = json.loads(
                    response.read(1_000_000) or b"{}"
                )

            fields = (data.get("model") or {}).get("input_fields")

            if (
                isinstance(fields, list)
                and fields
                and all(f in ALL_FIELDS for f in fields)
                and set(TX_FIELDS) <= set(fields)
            ):
                self.fields = tuple(
                    f for f in ALL_FIELDS if f in fields
                )
            else:
                log.warning(
                    "Model input fields unavailable; using base fields."
                )

        except urllib.error.HTTPError as error:
            log.warning(
                "Model field discovery returned HTTP %s; using base fields.",
                error.code,
            )

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            AttributeError,
        ):
            log.warning(
                "Model field discovery unavailable; using base fields."
            )

        return self.fields

    def _post(
        self,
        path: str,
        body: bytes,
        ctype: str,
    ) -> tuple[int, dict]:
        """Send an authenticated POST request."""
        req = urllib.request.Request(
            self.base + path,
            data=body,
            method="POST",
            headers=self._headers(ctype),
        )

        try:
            with urllib.request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                return (
                    response.status,
                    json.loads(
                        response.read(5_000_000) or b"{}"
                    ),
                )

        except urllib.error.HTTPError as error:
            try:
                data = json.loads(
                    error.read(100_000) or b"{}"
                )
            except ValueError:
                data = {}

            retry_after = (
                error.headers.get("Retry-After")
                if error.headers
                else None
            )

            return error.code, {
                **data,
                "_retry_after": retry_after,
            }

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ):
            return 0, {}

    def _with_retry(
        self,
        path: str,
        body: bytes,
        ctype: str,
    ) -> tuple[int, dict]:
        for attempt in range(self.max_retries + 1):
            status, data = self._post(
                path,
                body,
                ctype,
            )

            if (
                status != 429
                or attempt == self.max_retries
                or self.stop.is_set()
            ):
                return status, data

            try:
                wait = min(
                    float(data.get("_retry_after") or 1),
                    30.0,
                )
            except (ValueError, TypeError):
                wait = 1.0

            log.warning(
                "Rate limited; retrying in %.0f seconds.",
                wait,
            )
            self.sleep(wait)

        return status, data

    def _classify(
        self,
        status: int,
        data: dict,
    ) -> str:
        if status == 200:
            return (
                "scored_flagged"
                if data.get("flagged")
                else "scored_ok"
            )

        if status == 0:
            return "api_unreachable"

        error = data.get("error")
        code = (
            error.get("code")
            if isinstance(error, dict)
            else None
        )

        return f"api_{status}_{code or 'error'}"

    def send_one(
        self,
        row: dict,
        seq: int,
    ) -> None:
        """Send present model fields and log API validation failures."""
        body = json.dumps(
            {
                key: row[key]
                for key in self.fields
                if key in row
            }
        ).encode()

        status, data = self._with_retry(
            "/predict",
            body,
            "application/json",
        )

        # Diagnostic logging: exposes the actual API rejection reason.
        if status != 200:
            log.error(
                "API prediction failed: sequence=%s status=%s response=%s",
                seq,
                status,
                data,
            )

        self.rec.record(
            self._classify(status, data),
            seq,
            status,
        )

    def send_batch(
        self,
        rows: list[dict],
        first_seq: int,
    ) -> None:
        """Send batch transactions to FastAPI as CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(self.fields)

        for row in rows:
            writer.writerow(
                [row.get(key, "") for key in self.fields]
            )

        status, data = self._with_retry(
            "/predict/batch",
            buf.getvalue().encode(),
            "text/csv",
        )

        if status == 200:
            results = data.get("results", [])

            for index, result in enumerate(results):
                self.rec.record(
                    (
                        "scored_flagged"
                        if result.get("flagged")
                        else "scored_ok"
                    ),
                    first_seq + index,
                    200,
                )

            if len(results) != len(rows):
                log.error(
                    "Batch response mismatch: sent %d, received %d.",
                    len(rows),
                    len(results),
                )
        else:
            log.error(
                "API batch failed: status=%s response=%s",
                status,
                data,
            )

            for index in range(len(rows)):
                self.rec.record(
                    self._classify(status, data),
                    first_seq + index,
                    status,
                )


class KafkaSink:
    """Publishes validated transaction messages to Kafka."""

    def __init__(
        self,
        bootstrap: str,
        topic: str,
        recorder: OutcomeRecorder,
        producer=None,
        fields: tuple = TX_FIELDS,
    ):
        self.topic = topic
        self.rec = recorder
        self.fields = fields

        if producer is None:
            try:
                from confluent_kafka import Producer
            except ImportError as error:
                raise RuntimeError(
                    "confluent-kafka is not installed. "
                    "Install it using pip install confluent-kafka."
                ) from error

            producer = Producer(
                {
                    "bootstrap.servers": bootstrap,
                    "client.id": "aiml01-replay",
                    "socket.timeout.ms": 10000,
                    "message.timeout.ms": 30000,
                }
            )

        self.producer = producer

    def _on_delivery(
        self,
        error,
        message,
    ) -> None:
        if error is not None:
            self.rec.record("kafka_delivery_failed")
            log.error(
                "Kafka message delivery failed: %s",
                error,
            )
        else:
            self.rec.record("published")

    def send_one(
        self,
        row: dict,
        seq: int,
    ) -> None:
        try:
            value = encode_message(
                row,
                seq,
                self.fields,
            )
        except Exception:
            self.rec.record(
                "invalid_row_skipped",
                seq,
            )
            return

        try:
            self.producer.produce(
                self.topic,
                value=value,
                key=str(row["nameOrig"]).encode(),
                callback=self._on_delivery,
            )
            self.producer.poll(0)

        except BufferError:
            log.warning(
                "Kafka producer queue full; waiting for delivery."
            )
            self.producer.poll(1)

            self.producer.produce(
                self.topic,
                value=value,
                key=str(row["nameOrig"]).encode(),
                callback=self._on_delivery,
            )

        except Exception:
            self.rec.record(
                "kafka_publish_error",
                seq,
            )
            raise

    def flush(self) -> None:
        """Wait for Kafka deliveries."""
        log.info("Waiting for Kafka message delivery...")

        remaining = self.producer.flush(30)

        if remaining:
            raise RuntimeError(
                f"Kafka delivery incomplete: {remaining} "
                "message(s) pending after 30 seconds."
            )

        log.info("Kafka producer flush completed.")


def consume_loop(
    consumer,
    sink: ApiSink,
    rec: OutcomeRecorder,
    stop: threading.Event,
    poll_timeout: float = 1.0,
    max_messages: int | None = None,
    fields: tuple | None = None,
) -> int:
    """Consume, validate, score, and commit Kafka messages."""
    handled = 0

    while (
        not stop.is_set()
        and (
            max_messages is None
            or handled < max_messages
        )
    ):
        msg = consumer.poll(poll_timeout)

        if msg is None:
            continue

        error = msg.error()

        if error:
            rec.record("kafka_error")
            log.error("Kafka error received.")
            continue

        try:
            seq, transaction = decode_message(
                msg.value(),
                fields or sink.fields,
            )

        except MessageError as error:
            rec.record(
                f"malformed_{error.code}"
            )

            consumer.commit(
                msg,
                asynchronous=False,
            )

            handled += 1
            continue

        sink.send_one(
            transaction,
            seq if seq is not None else handled,
        )

        consumer.commit(
            msg,
            asynchronous=False,
        )

        handled += 1

    return handled