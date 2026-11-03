"""Show the chronological split WITHOUT training:  python -m ml.splits --data <csv> [--nrows N] [--json]
Reports exact step boundaries, row counts and class balance per split, and flags any split that lacks both classes."""
from __future__ import annotations

import argparse
import json
import sys

from ml.pipeline.schema import SchemaError, TARGET
from ml.pipeline.split import time_split
from ml.train import DatasetError, load_dataset


def describe_splits(df, train_frac: float = 0.6, val_frac: float = 0.2) -> dict:
    tr, va, te, bounds = time_split(df, train_frac, val_frac)
    out = {"boundaries": bounds, "splits": {}}
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        pos = int(part[TARGET].sum())
        out["splits"][name] = {"rows": int(len(part)), "steps": [int(part["step"].min()), int(part["step"].max())],
                               "distinct_steps": int(part["step"].nunique()), "fraud_rows": pos,
                               "fraud_rate": float(part[TARGET].mean()),
                               "usable": bool(0 < pos < len(part))}
    out["all_splits_usable"] = all(s["usable"] for s in out["splits"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Describe the time-ordered train/validation/test split.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        d = describe_splits(load_dataset(a.data, a.nrows), a.train_frac, a.val_frac)
    except (DatasetError, SchemaError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        for n, s in d["splits"].items():
            flag = "" if s["usable"] else "   <-- UNUSABLE (needs both classes)"
            print(f"{n:11s} steps {s['steps'][0]:>4}-{s['steps'][1]:<4} rows {s['rows']:>9,} fraud {s['fraud_rows']:>7,} "
                  f"({s['fraud_rate']:.4%}){flag}")
        print("all splits usable:", d["all_splits_usable"])
    return 0 if d["all_splits_usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
