# Streaming and replay

Kafka is **optional**. The Kafka-free path replays a dataset file straight into the API.

| Component | Status |
|---|---|
| `python -m streaming api` (CSV → API, single or `--api-mode batch`) | Tested against a real local HTTP server (stdlib fake API) |
| Pacing, graceful stop, 429 `Retry-After` handling, outcome recording | Tested |
| Message validation (`streaming/messages.py`) | Tested |
| `python -m streaming produce` / `consume` (Kafka via `confluent-kafka`) | Logic tested with **fake** Kafka clients only. **No broker was started; the real library was not exercised.** |

## Replay semantics
Rows come from `ml.train.load_dataset` (same validation as training; only the needed columns are read), sorted by `step`.
`--speed S` = simulated seconds per real second (1 step = 3600 s), so `--speed 3600` replays one dataset hour per real
second; `--speed 0` (default) means no delay. Nothing is fabricated: rows are exactly those in your file.
The API limits `/predict` to 60 requests/min by default; the replay client waits for `Retry-After` on 429. For fast replays
use `--api-mode batch` or raise `RATE_LIMIT_PREDICT_PER_MIN` for the demo.

## Commands (PowerShell, API already running)
```
python -m streaming api --data ml\data\raw\<file>.csv --nrows 2000 --api-mode batch --batch-size 200
python -m streaming api --data ml\data\raw\<file>.csv --limit 200 --speed 3600
# Kafka (only if you installed confluent-kafka and run a broker yourself):
python -m streaming produce --data ml\data\raw\<file>.csv --bootstrap localhost:9092 --topic transactions
python -m streaming consume --bootstrap localhost:9092 --topic transactions
```
`Ctrl+C` stops cleanly after the current record. Outcomes (counts + a content-free JSONL) go to `streaming_outcomes/`
(git-ignored); the dashboard reads `last_run_summary.json`. Consumer semantics: validate → score via API → commit
(at-least-once); malformed messages are counted and committed so they cannot block a partition. A replay is a test harness,
not production traffic, and nothing here is described as a live Kafka deployment.
