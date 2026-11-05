"""CLI for CSV replay, Kafka production, and Kafka-to-API consumption."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from ml.pipeline.schema import FEATURE_SET_INPUTS, FEATURE_SETS
from streaming.outcomes import OutcomeRecorder
from streaming.replay import (
    ApiSink,
    KafkaSink,
    consume_loop,
    load_rows,
    replay,
)

log = logging.getLogger("fraudapp.streaming")


def install_signal_handlers(stop: threading.Event) -> None:
    def handler(signum, frame):
        log.info("Shutdown requested; finishing current record.")
        stop.set()

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        signal.signal(sig, handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m streaming")

    ap.add_argument("mode", choices=["api", "produce", "consume"])
    ap.add_argument("--data", help="CSV path (api/produce modes)")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument(
        "--feature-set",
        default="auto",
        choices=("auto",) + FEATURE_SETS,
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--speed",
        type=float,
        default=float(os.environ.get("REPLAY_SPEED_MULTIPLIER", "0")),
    )
    ap.add_argument(
        "--api-url",
        default=os.environ.get(
            "API_URL",
            "http://127.0.0.1:8000/api/v1",
        ),
    )
    ap.add_argument("--api-mode", choices=["single", "batch"], default="single")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument(
        "--bootstrap",
        default=os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS",
            "127.0.0.1:9094",
        ),
    )
    ap.add_argument(
        "--topic",
        default=os.environ.get("KAFKA_TOPIC", "transactions"),
    )
    ap.add_argument("--group", default="aiml01-consumer")
    ap.add_argument(
        "--outcomes-dir",
        default=os.environ.get("STREAM_OUTCOMES_DIR", "streaming_outcomes"),
    )

    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stop = threading.Event()
    install_signal_handlers(stop)

    rec = OutcomeRecorder(Path(a.outcomes_dir), a.mode)

    try:
        if a.mode in ("api", "produce"):
            if not a.data:
                print("ERROR: --data is required", file=sys.stderr)
                return 2

            sink = (
                ApiSink(a.api_url, rec, stop=stop)
                if a.mode == "api"
                else None
            )

            if a.feature_set != "auto":
                fields = tuple(FEATURE_SET_INPUTS[a.feature_set])
            elif sink is not None:
                fields = sink.discover_fields()
            else:
                fields = tuple(FEATURE_SET_INPUTS["base"])

            if sink is not None:
                sink.fields = fields

            log.info("Sending fields: %s", ", ".join(fields))

            rows = load_rows(a.data, a.nrows, fields)

            if a.mode == "api":
                if a.api_mode == "single":
                    n = replay(
                        rows,
                        sink.send_one,
                        speed=a.speed,
                        stop=stop,
                        limit=a.limit,
                    )
                else:
                    buffer = []
                    start = 0

                    def emit(row, seq):
                        nonlocal start
                        buffer.append(row)

                        if len(buffer) >= a.batch_size:
                            sink.send_batch(buffer[:], start)
                            start += len(buffer)
                            buffer.clear()

                    n = replay(
                        rows,
                        emit,
                        speed=a.speed,
                        stop=stop,
                        limit=a.limit,
                    )

                    if buffer:
                        sink.send_batch(buffer, start)

            else:
                kafka_sink = KafkaSink(
                    a.bootstrap,
                    a.topic,
                    rec,
                    fields=fields,
                )

                n = replay(
                    rows,
                    kafka_sink.send_one,
                    speed=a.speed,
                    stop=stop,
                    limit=a.limit,
                )

                kafka_sink.flush()

            log.info("Emitted %d rows", n)

        else:
            try:
                from confluent_kafka import Consumer
            except ImportError:
                print(
                    "ERROR: confluent-kafka is not installed.",
                    file=sys.stderr,
                )
                return 2

            consumer = Consumer(
                {
                    "bootstrap.servers": a.bootstrap,
                    "group.id": a.group,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,
                }
            )

            consumer.subscribe([a.topic])

            try:
                csink = ApiSink(a.api_url, rec, stop=stop)

                if a.feature_set != "auto":
                    csink.fields = tuple(
                        FEATURE_SET_INPUTS[a.feature_set]
                    )
                else:
                    csink.discover_fields()

                log.info("Kafka consumer started.")
                log.info("Topic: %s", a.topic)
                log.info("API endpoint: %s", a.api_url)
                log.info(
                    "Maximum messages: %s",
                    a.limit if a.limit is not None else "unlimited",
                )

                handled = consume_loop(
                    consumer,
                    csink,
                    rec,
                    stop,
                    max_messages=a.limit,
                )

                log.info("Consumer handled %d messages.", handled)

            finally:
                consumer.close()

    except Exception as exc:
        log.exception("Streaming operation failed.")
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    finally:
        print(rec.close())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())