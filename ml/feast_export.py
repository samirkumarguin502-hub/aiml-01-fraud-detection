"""python -m ml.feast_export --data <csv> --out feature_repo/data/orig_history.parquet"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml.feast_support import FeastSupportError, write_feature_table
from ml.train import DatasetError, load_dataset
from ml.pipeline.schema import SchemaError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="feature_repo/data/orig_history.parquet")
    ap.add_argument("--nrows", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        path = write_feature_table(load_dataset(a.data, a.nrows), Path(a.out))
    except (DatasetError, SchemaError, FeastSupportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
