"""Monitoring CLI.

Examples:
    python -m monitoring status --db fraud_platform.db \
        --artifact ml/artifacts/<run>/primary

    python -m monitoring evidently --data <csv> \
        --artifact ml/artifacts/<run>/primary \
        --db fraud_platform.db --out report.html

Evidently is optional. Primary PSI monitoring does not depend on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backend.app.config import Settings
from backend.app.db import Repository
from backend.app.model_service import ModelService
from backend.app.service import FraudService


def _service(db: str, artifact: str | None) -> FraudService:
    settings = Settings(
        database_path=Path(db),
        model_dir=Path(artifact) if artifact else None,
    )

    repo = Repository(settings.database_path)
    repo.init_schema()

    model = ModelService(settings.model_dir)
    model.load()

    return FraudService(settings, repo, model)


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _build_evidently_report(
    svc: FraudService,
    data_path: str,
    output_path: str,
) -> dict:
    """Build an optional Evidently report using the loaded model contract."""
    from ml.pipeline.features import build_features
    from ml.pipeline.split import time_split
    from monitoring.evidently_adapter import build_html_report

    loaded = svc.model.loaded

    if loaded is None:
        return {"status": "unavailable: model_not_loaded"}

    contract = loaded.contract
    input_columns = list(contract.input_columns)

    if not input_columns:
        return {"status": "failed: empty_model_input_contract"}

    dataset_path = Path(data_path)

    if not dataset_path.is_file():
        return {"status": "failed: dataset_file_not_found"}

    # Read the original CSV so the artifact's feature contract can select
    # the correct raw columns, including optional balance columns.
    try:
        df = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return {"status": f"failed: dataset_read_{type(exc).__name__}"}

    missing = [column for column in input_columns if column not in df.columns]

    if missing:
        return {
            "status": "failed: reference_dataset_missing_model_inputs",
            "missing_columns": missing,
        }

    try:
        train, _, _, _ = time_split(df)
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": f"failed: time_split_{type(exc).__name__}"}

    if train.empty:
        return {"status": "insufficient_reference_data"}

    # Build features on the full chronological dataset so historical
    # features retain their intended earlier-transaction context.
    try:
        reference_features = build_features(df, contract)
        reference_features = reference_features.loc[train.index]
        reference_features = reference_features.drop(
            columns=["day_index"],
            errors="ignore",
        )

        reference = reference_features.sample(
            n=min(5000, len(reference_features)),
            random_state=0,
        )
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": f"failed: reference_features_{type(exc).__name__}"}

    # Use exactly the raw input columns required by the loaded artifact.
    current_rows = svc.repo.monitoring_rows()

    if current_rows is None or current_rows.empty:
        return {
            "status": "insufficient_data",
            "reason": "no_current_monitoring_rows",
        }

    missing_current = [
        column
        for column in input_columns
        if column not in current_rows.columns
    ]

    if missing_current:
        return {
            "status": "failed: current_data_missing_model_inputs",
            "missing_columns": missing_current,
        }

    try:
        current = build_features(
            current_rows.loc[:, input_columns].copy(),
            contract,
        )
        current = current.drop(
            columns=["day_index"],
            errors="ignore",
        )
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": f"failed: current_features_{type(exc).__name__}"}

    return build_html_report(
        reference=reference,
        current=current,
        out_path=Path(output_path),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monitoring",
        description="Fraud platform monitoring utilities.",
    )

    parser.add_argument(
        "mode",
        choices=["status", "evidently"],
        help="Monitoring operation to execute.",
    )
    parser.add_argument(
        "--db",
        default="fraud_platform.db",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="Trained model artifact directory.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Reference dataset CSV for Evidently reporting.",
    )
    parser.add_argument(
        "--out",
        default="evidently_report.html",
        help="Output HTML report path.",
    )

    args = parser.parse_args(argv)

    try:
        service = _service(args.db, args.artifact)

        if args.mode == "status":
            _print_json(service.monitoring_status())
            return 0

        if not args.data:
            print(
                "ERROR: evidently mode requires --data.",
                file=sys.stderr,
            )
            return 2

        if not args.artifact or not service.model.ready:
            print(
                "ERROR: evidently mode requires a valid --artifact.",
                file=sys.stderr,
            )
            return 2

        result = _build_evidently_report(
            svc=service,
            data_path=args.data,
            output_path=args.out,
        )

        _print_json(result)

        return 0 if result.get("status") == "generated" else 1

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())