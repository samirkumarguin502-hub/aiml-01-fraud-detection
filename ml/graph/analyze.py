"""Analyze the PaySim sender-to-receiver account graph.

Usage:
    python -m ml.graph.analyze --data <csv>
    python -m ml.graph.analyze --data <csv> --max-step 100
    python -m ml.graph.analyze --data <csv> --train-gnn
    python -m ml.graph.analyze --data <csv> --train-gnn --gnn-epochs 100

All graph operations use the same filtered dataset when --max-step is set.
"""
from __future__ import annotations

import argparse
import json
import sys

from ml.graph.build import (
    GraphInputError,
    evaluate_graph_baseline,
    graph_status,
    graph_summary,
    train_gnn,
)
from ml.pipeline.schema import SchemaError
from ml.train import DatasetError, load_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze PaySim account-transfer graph."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--nrows", type=int, default=None)

    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only use transactions up to this step.",
    )

    parser.add_argument(
        "--train-gnn",
        action="store_true",
        help="Train optional CPU GraphSAGE on the filtered training graph.",
    )

    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--gnn-out",
        default=None,
        help="Optional path for the trained PyG model state.",
    )

    args = parser.parse_args(argv)

    try:
        if args.nrows is not None and args.nrows < 1:
            raise ValueError("--nrows must be greater than zero.")

        if args.max_step is not None and args.max_step < 0:
            raise ValueError("--max-step cannot be negative.")

        if args.gnn_epochs < 1:
            raise ValueError("--gnn-epochs must be at least 1.")

        # Load the dataset once.
        df = load_dataset(args.data, args.nrows)

        # Apply the same chronological boundary to every graph operation.
        if args.max_step is not None:
            df = df.loc[df["step"] <= args.max_step].copy()

        if df.empty:
            raise GraphInputError(
                "No transactions remain after applying --max-step."
            )

        # Validate the common filtered input before generating output.
        summary = graph_summary(df)

        output = {
            "status": graph_status(),
            "data_scope": {
                "rows": int(len(df)),
                "minimum_step": int(df["step"].min()),
                "maximum_step": int(df["step"].max()),
                "max_step_filter": args.max_step,
                "nrows_limit": args.nrows,
            },
            "summary": summary,
            "baseline_evaluation": evaluate_graph_baseline(df),
        }

        if args.train_gnn:
            output["gnn_training"] = train_gnn(
                df,
                epochs=args.gnn_epochs,
                out_path=args.gnn_out,
            )

    except (
        DatasetError,
        SchemaError,
        GraphInputError,
        ValueError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())