"""Processing outcomes: counts and a content-free JSONL log. Never records transaction fields."""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


class OutcomeRecorder:
    def __init__(self, out_dir: Path | None, mode: str):
        self.counts: Counter = Counter()
        self.mode, self.started = mode, datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._t0 = time.monotonic()
        self._fh = None
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = (self.out_dir / "outcomes.jsonl").open("a", encoding="utf-8")

    def record(self, outcome: str, seq: int | None = None, http_status: int | None = None) -> None:
        self.counts[outcome] += 1
        if self._fh:
            self._fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       "seq": seq, "outcome": outcome, "http_status": http_status}) + "\n")

    def summary(self, finished: bool = True) -> dict:
        total = sum(self.counts.values())
        return {"mode": self.mode, "started_utc": self.started, "finished": finished,
                "elapsed_seconds": round(time.monotonic() - self._t0, 2), "processed": total,
                "outcomes": dict(self.counts),
                "note": "Counts only; transaction content is never recorded. A replay is a test harness, "
                        "not live production traffic."}

    def close(self) -> dict:
        s = self.summary()
        if self._fh:
            self._fh.close()
        if self.out_dir:
            (self.out_dir / "last_run_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        return s
