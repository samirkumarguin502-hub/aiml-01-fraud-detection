"""Optional Evidently HTML drift report.

Evidently is an optional dependency. The platform's primary PSI monitoring
does not depend on it.

Supports the newer Evidently API (0.7+) and the legacy API (0.4-0.6).
Failures are returned as status dictionaries instead of breaking monitoring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def evidently_version() -> str | None:
    """Return installed Evidently version, or None if unavailable."""
    try:
        import evidently

        return str(getattr(evidently, "__version__", "unknown"))
    except ImportError:
        return None


def _validate_data(reference: Any, current: Any) -> str | None:
    """Validate input datasets before calling Evidently."""
    try:
        import pandas as pd
    except ImportError:
        return "failed: pandas_not_installed"

    if not isinstance(reference, pd.DataFrame):
        return "failed: reference_must_be_dataframe"

    if not isinstance(current, pd.DataFrame):
        return "failed: current_must_be_dataframe"

    if reference.empty or current.empty:
        return "insufficient_data"

    if len(reference.columns) == 0 or len(current.columns) == 0:
        return "failed: no_columns"

    if not reference.columns.is_unique or not current.columns.is_unique:
        return "failed: duplicate_columns"

    if list(reference.columns) != list(current.columns):
        return "failed: column_schema_mismatch"

    return None


def build_html_report(
    reference,
    current,
    out_path: Path,
) -> dict:
    """Generate an optional Evidently HTML data-drift report.

    Args:
        reference: Reference/training DataFrame.
        current: Current/production DataFrame.
        out_path: Destination HTML file path.

    Returns:
        Status dictionary. This function does not raise for optional
        Evidently failures.
    """
    ver = evidently_version()

    if ver is None:
        return {"status": "unavailable: evidently not installed"}

    validation_error = _validate_data(reference, current)

    if validation_error == "insufficient_data":
        return {"status": "insufficient_data", "version": ver}

    if validation_error is not None:
        return {"status": validation_error, "version": ver}

    try:
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # New API: Evidently 0.7+
        try:
            from evidently import DataDefinition, Dataset, Report
            from evidently.presets import DataDriftPreset
        except ImportError:
            DataDefinition = Dataset = Report = None
            DataDriftPreset = None

        if all(
            item is not None
            for item in (DataDefinition, Dataset, Report, DataDriftPreset)
        ):
            reference_dataset = Dataset.from_pandas(
                reference,
                data_definition=DataDefinition(),
            )

            current_dataset = Dataset.from_pandas(
                current,
                data_definition=DataDefinition(),
            )

            report = Report([DataDriftPreset()])
            snapshot = report.run(current_dataset, reference_dataset)
            snapshot.save_html(str(destination))

            api = "new"

        else:
            # Legacy API: Evidently 0.4-0.6
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference,
                current_data=current,
            )
            report.save_html(str(destination))

            api = "legacy"

        if not destination.is_file() or destination.stat().st_size == 0:
            return {
                "status": "failed: html_output_not_created",
                "version": ver,
                "api": api,
            }

        return {
            "status": "generated",
            "version": ver,
            "api": api,
            "path": destination.name,
        }

    except Exception as exc:
        return {
            "status": f"failed: {type(exc).__name__}",
            "version": ver,
        }