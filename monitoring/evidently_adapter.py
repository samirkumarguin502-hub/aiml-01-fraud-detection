"""OPTIONAL Evidently HTML drift report. NOT verified: evidently is not installed in the authoring
sandbox, and its API changed across releases. This tries the newer API first, then the legacy one,
and returns a status instead of raising. The platform's own PSI monitoring does not depend on it."""
from __future__ import annotations

from pathlib import Path


def evidently_version() -> str | None:
    try:
        import evidently
        return getattr(evidently, "__version__", "unknown")
    except ImportError:
        return None


def build_html_report(reference, current, out_path: Path) -> dict:
    ver = evidently_version()
    if ver is None:
        return {"status": "unavailable: evidently not installed"}
    if len(reference) == 0 or len(current) == 0:
        return {"status": "insufficient_data", "version": ver}
    try:
        try:  # newer API (0.7+)
            from evidently import DataDefinition, Dataset, Report
            from evidently.presets import DataDriftPreset
            ref = Dataset.from_pandas(reference, data_definition=DataDefinition())
            cur = Dataset.from_pandas(current, data_definition=DataDefinition())
            snap = Report([DataDriftPreset()]).run(cur, ref)
            snap.save_html(str(out_path))
        except ImportError:  # legacy API (0.4 - 0.6)
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=reference, current_data=current)
            rep.save_html(str(out_path))
        return {"status": "generated", "version": ver, "path": out_path.name}
    except Exception as e:  # noqa: BLE001
        return {"status": f"failed: {type(e).__name__}", "version": ver}
