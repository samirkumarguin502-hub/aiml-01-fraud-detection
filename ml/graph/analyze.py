"""python -m ml.graph.analyze --data <csv> [--max-step N]   (prints JSON built from the real file)"""
from __future__ import annotations

import argparse
import json
import sys

from ml.graph.build import GraphInputError, evaluate_graph_baseline, graph_status, graph_summary
from ml.pipeline.schema import SchemaError
from ml.train import DatasetError, load_dataset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None, help="Only use edges up to this step (e.g. train range)")
    a = ap.parse_args(argv)
    try:
        df = load_dataset(a.data, a.nrows)
        sub = df if a.max_step is None else df[df["step"] <= a.max_step]
        out = {"status": graph_status(), "summary": graph_summary(sub), "baseline_evaluation": evaluate_graph_baseline(df)}
    except (DatasetError, SchemaError, GraphInputError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
