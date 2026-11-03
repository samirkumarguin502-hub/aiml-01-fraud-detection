"""CLI.
  python -m monitoring status    --db fraud_platform.db --artifact ml/artifacts/<run>/primary
      Prints the same monitoring report the API serves, from the SQLite file + artifact.
  python -m monitoring evidently --data <csv> --artifact <dir> --db <sqlite> --out report.html
      Optional HTML drift report (needs `pip install evidently`; NOT verified in the authoring sandbox).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.config import Settings
from backend.app.db import Repository
from backend.app.model_service import ModelService
from backend.app.service import FraudService


def _service(db: str, artifact: str | None) -> FraudService:
    s = Settings(database_path=Path(db), model_dir=Path(artifact) if artifact else None)
    repo = Repository(s.database_path)
    repo.init_schema()
    m = ModelService(s.model_dir)
    m.load()
    return FraudService(s, repo, m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m monitoring")
    ap.add_argument("mode", choices=["status", "evidently"])
    ap.add_argument("--db", default="fraud_platform.db")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="evidently_report.html")
    a = ap.parse_args(argv)
    try:
        svc = _service(a.db, a.artifact)
        if a.mode == "status":
            print(json.dumps(svc.monitoring_status(), indent=2, default=str))
            return 0
        if not a.data or not svc.model.ready:
            print("ERROR: evidently mode needs --data and a valid --artifact", file=sys.stderr)
            return 2
        from ml.pipeline.features import build_features
        from ml.pipeline.split import time_split
        from ml.train import load_dataset
        from monitoring.evidently_adapter import build_html_report
        df = load_dataset(a.data).reset_index(drop=True)
        train, _, _, _ = time_split(df)
        contract = svc.model.loaded.contract
        ref = build_features(df, contract).loc[train.index].drop(columns=["day_index"]).sample(
            n=min(5000, len(train)), random_state=0)
        cur_rows = svc.repo.monitoring_rows()
        cur = build_features(cur_rows[["step", "type", "amount", "nameOrig", "nameDest"]], contract).drop(columns=["day_index"])
        print(json.dumps(build_html_report(ref, cur, Path(a.out))))
        return 0
    except Exception as e:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
