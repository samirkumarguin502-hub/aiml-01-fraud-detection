"""Reproducible training + evaluation for AIML-01.

    python -m ml.train --data ml/data/raw/<paysim>.csv --model xgboost

Protocol:
- Time split on step (train < validation < test).
- Early stopping and threshold selection use VALIDATION only.
- Test set is scored after the threshold is fixed.
- Test data is never used to choose between models.
- Model selection uses validation PR-AUC.
- Only feature columns and target are loaded from CSV.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.pipeline import plots
from ml.pipeline.evaluate import (
    classification_report_dict,
    compute_metrics,
    curve_points,
    ranking_metrics,
    select_threshold,
    threshold_table,
)
from ml.pipeline.features import FeatureContract, build_features
from ml.pipeline.model_io import _jsonable, load_artifact, save_artifact
from ml.pipeline.reference import build_reference_profile
from ml.pipeline.models import (
    MODEL_KINDS,
    MissingDependencyError,
    ScoreModel,
    _import_xgboost,
)
from ml.pipeline.resources import environment_info, peak_memory_mb
from ml.pipeline.schema import (
    FEATURE_SET_INPUTS,
    FEATURE_SETS,
    REQUIRED_COLUMNS,
    TARGET,
    SchemaError,
    validate_schema,
)
from ml.pipeline.split import time_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ml" / "artifacts"


class DatasetError(RuntimeError):
    """Dataset path missing/unreadable/not a usable CSV."""


class TrainingError(RuntimeError):
    """Data cannot support a valid train/validation/test protocol."""


def sha256_file(path: Path) -> str:
    from ml.eda.inspect_dataset import sha256_of
    return sha256_of(path)


def load_dataset(
    path: str | Path,
    nrows: int | None = None,
    feature_set: str = "base",
) -> pd.DataFrame:
    """Read only feature columns and target from the CSV."""
    if feature_set not in FEATURE_SET_INPUTS:
        raise DatasetError(
            f"Unknown feature set {feature_set!r}; "
            f"choose from {FEATURE_SETS}"
        )

    if nrows is not None and nrows < 1:
        raise DatasetError("nrows must be a positive integer.")

    needed = REQUIRED_COLUMNS + [
        c for c in FEATURE_SET_INPUTS[feature_set]
        if c not in REQUIRED_COLUMNS
    ]

    path = Path(path)

    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}. "
            "See docs/dataset.md for how to obtain it."
        )

    if not path.is_file():
        raise DatasetError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise DatasetError(
            f"Dataset must be a .csv file, got '{path.suffix}'."
        )

    if path.stat().st_size == 0:
        raise DatasetError(f"Dataset file is empty: {path.name}")

    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        OSError,
    ) as exc:
        raise DatasetError(
            f"Could not parse CSV header of {path.name}: "
            f"{type(exc).__name__}"
        ) from exc

    if len(header) != len(set(header)):
        raise DatasetError("CSV contains duplicate column names.")

    missing = [c for c in needed if c not in header]

    if missing:
        raise SchemaError(
            f"Missing required columns {missing}. Found: {header}"
        )

    try:
        df = pd.read_csv(
            path,
            usecols=needed,
            nrows=nrows,
        )
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ) as exc:
        raise DatasetError(
            f"Could not parse {path.name}: {type(exc).__name__}"
        ) from exc

    if df.empty:
        raise DatasetError("CSV contains no transaction rows.")

    return df.reset_index(drop=True)


def _flat(prefix: str, d: dict) -> dict:
    """Flatten numeric metrics for MLflow."""
    return {
        f"{prefix}{k}": v
        for k, v in d.items()
        if isinstance(v, (int, float, np.integer, np.floating))
        and v is not None
        and np.isfinite(v)
    }


def _verify_reload(
    out_dir: Path,
    model,
    contract,
    X_sample,
) -> dict:
    """Compare in-memory and reloaded model predictions."""
    lm = load_artifact(out_dir)

    reference_scores = model.predict_score(X_sample)
    loaded_scores = lm.model.predict_score(X_sample)

    diff = (
        float(np.max(np.abs(reference_scores - loaded_scores)))
        if len(X_sample)
        else 0.0
    )

    schema_match = (
        lm.contract.feature_names == contract.feature_names
        and lm.contract.feature_set == contract.feature_set
    )

    return {
        "rows": int(len(X_sample)),
        "max_abs_diff": diff,
        "tolerance": 1e-6,
        "feature_schema_match": bool(schema_match),
        "passed": bool(diff <= 1e-6 and schema_match),
    }


def _train_one(
    kind,
    df,
    X,
    tr,
    va,
    te,
    seed,
    min_precision,
    params,
    out_dir,
    provenance,
    contract,
    run_id,
):
    """Train one model and evaluate it using validation-selected threshold."""
    y = df[TARGET]

    model = ScoreModel(
        kind,
        params=params,
        seed=seed,
    )

    t0 = time.perf_counter()

    model.fit(
        X.loc[tr.index],
        y.loc[tr.index],
        X.loc[va.index],
        y.loc[va.index],
    )

    fit_seconds = round(time.perf_counter() - t0, 2)

    # Scores for validation and test.
    s_val = model.predict_score(X.loc[va.index])
    s_test = model.predict_score(X.loc[te.index])

    # Threshold selection uses validation only.
    thr = select_threshold(
        y.loc[va.index],
        s_val,
        min_precision,
    )

    val_metrics = compute_metrics(
        y.loc[va.index],
        s_val,
        thr["threshold"],
    )

    test_metrics = compute_metrics(
        y.loc[te.index],
        s_test,
        thr["threshold"],
    )

    # Test metrics are reporting-only.
    table = threshold_table(
        y.loc[te.index],
        s_test,
    )

    report = classification_report_dict(
        y.loc[te.index],
        s_test,
        thr["threshold"],
    )

    curves = curve_points(
        y.loc[te.index],
        s_test,
    )

    importance = model.feature_importance(
        contract.feature_names
    )

    training = {
        "fit_seconds": fit_seconds,
        "peak_memory_mb": peak_memory_mb(),
        **environment_info(),
    }

    # IMPORTANT:
    # Feature distributions come from TRAIN.
    # Prediction-score distribution comes from VALIDATION.
    # Their row counts do not need to match.
    reference_profile = build_reference_profile(
        X.loc[tr.index],
        s_val,
    )

    meta = save_artifact(
        out_dir,
        model,
        contract,
        threshold=thr,
        metrics={
            "validation": val_metrics,
            "test": test_metrics,
            "test_threshold_table": table,
            "test_classification_report": report,
            "test_curves": curves,
            "feature_importance": importance,
        },
        provenance=provenance,
        reference_profile=reference_profile,
        extra={
            "model_version": run_id,
            "feature_set": contract.feature_set,
            "training": training,
        },
    )

    reload_check = _verify_reload(
        out_dir,
        model,
        contract,
        X.loc[te.index[:2000]],
    )

    if not reload_check["passed"]:
        raise TrainingError(
            f"Saved model failed the reload check ({reload_check})."
        )

    return {
        "kind": kind,
        "artifact_dir": out_dir.name,
        "model_sha256": meta["model_sha256"],
        "threshold": thr,
        "validation": val_metrics,
        "test": test_metrics,
        "test_threshold_table": table,
        "classification_report": report,
        "curves": curves,
        "feature_importance": importance,
        "params": meta["params"],
        "fit_seconds": fit_seconds,
        "reload_check": reload_check,
        "fit_info": model.fit_info,
        "test_scores": s_test,
    }


DEFAULT_GRID = {
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.3],
}


def _search(
    kind,
    grid,
    base_params,
    X,
    df,
    tr,
    va,
    seed,
):
    """Grid search scored on validation PR-AUC only."""
    y = df[TARGET]
    keys = sorted(grid)
    trials = []

    if not keys or any(not grid[k] for k in keys):
        raise TrainingError("Hyperparameter search grid cannot be empty.")

    for combo in itertools.product(*(grid[k] for k in keys)):
        cp = dict(zip(keys, combo))

        t0 = time.perf_counter()

        model = ScoreModel(
            kind,
            params={**(base_params or {}), **cp},
            seed=seed,
        )

        model.fit(
            X.loc[tr.index],
            y.loc[tr.index],
            X.loc[va.index],
            y.loc[va.index],
        )

        scores = model.predict_score(X.loc[va.index])

        pr_auc = ranking_metrics(
            y.loc[va.index],
            scores,
        )["pr_auc"]

        trials.append({
            "params": cp,
            "val_pr_auc": pr_auc,
            "fit_seconds": round(
                time.perf_counter() - t0,
                2,
            ),
        })

    valid_trials = [
        trial for trial in trials
        if trial["val_pr_auc"] is not None
    ]

    if not valid_trials:
        raise TrainingError(
            "Hyperparameter search produced no valid validation PR-AUC."
        )

    best = max(
        valid_trials,
        key=lambda trial: trial["val_pr_auc"],
    )

    return (
        {**(base_params or {}), **best["params"]},
        {
            "selection_metric": "validation PR-AUC",
            "trials": trials,
            "chosen": best["params"],
            "test_data_used": False,
        },
    )


def run_training(
    df: pd.DataFrame,
    *,
    kind: str = "xgboost",
    seed: int = 42,
    min_precision: float = 0.80,
    params: dict | None = None,
    out_root: Path = DEFAULT_OUT,
    run_id: str | None = None,
    data_label: str = "UNSPECIFIED",
    dataset_sha256: str | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    use_mlflow: bool = True,
    feature_set: str = "base",
    grid: dict | None = None,
    make_plots: bool = True,
) -> dict:
    """Run reproducible training, evaluation and artifact creation."""
    t_start = time.perf_counter()

    if kind not in MODEL_KINDS:
        raise TrainingError(
            f"Unsupported model kind {kind!r}. "
            f"Choose from {MODEL_KINDS}."
        )

    if not 0 < min_precision <= 1:
        raise TrainingError("min_precision must be in (0, 1].")

    if kind == "xgboost":
        _import_xgboost()

    if grid and kind == "logreg":
        raise TrainingError(
            "A hyperparameter search is not available for logreg."
        )

    random.seed(seed)
    np.random.seed(seed)

    if not isinstance(df, pd.DataFrame):
        raise TrainingError("Training input must be a DataFrame.")

    df = df.reset_index(drop=True)

    facts = validate_schema(
        df,
        feature_set=feature_set,
    )

    tr, va, te, steps = time_split(
        df,
        train_frac,
        val_frac,
    )

    # Each split must contain both classes.
    for name, part in (
        ("train", tr),
        ("validation", va),
        ("test", te),
    ):
        positives = int(part[TARGET].sum())

        if positives == 0 or positives == len(part):
            raise TrainingError(
                f"The {name} split has {positives} fraud rows "
                f"of {len(part)}; both classes are required. "
                "Use more data or different split fractions."
            )

    # Feature vocabulary is fitted on training data only.
    contract = FeatureContract.fit(
        tr,
        feature_set,
    )

    # History features use only strictly earlier steps.
    X = build_features(
        df,
        contract,
    )

    run_id = run_id or (
        f"{kind}-{feature_set}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        f"-seed{seed}"
    )

    if (
        not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        raise TrainingError(
            "run_id must be a simple directory name without path separators."
        )

    run_dir = Path(out_root) / run_id

    if run_dir.exists():
        raise TrainingError(
            f"Run directory already exists: {run_dir.name}; "
            "choose a new --run-id."
        )

    provenance = {
        "data_label": data_label,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "feature_set": feature_set,
        "split_steps": steps,
        "rows": {
            "train": len(tr),
            "validation": len(va),
            "test": len(te),
        },
        "schema_facts": facts,
    }

    search = None

    if grid:
        params, search = _search(
            kind,
            grid,
            params,
            X,
            df,
            tr,
            va,
            seed,
        )

    results = {}
    plots_written = []

    kinds = (
        ["logreg"]
        if kind == "logreg"
        else ["logreg", kind]
    )

    try:
        for model_kind in kinds:
            role = (
                "baseline"
                if model_kind == "logreg" and kind != "logreg"
                else "primary"
            )

            results[role] = _train_one(
                model_kind,
                df,
                X,
                tr,
                va,
                te,
                seed,
                min_precision,
                params if model_kind == kind else None,
                run_dir / role,
                provenance,
                contract,
                run_id,
            )

        if make_plots:
            primary = results["primary"]

            plots_written = plots.save_all(
                run_dir / "plots",
                label=data_label,
                test_metrics=primary["test"],
                curves=primary["curves"],
                baseline_curves=(
                    results.get("baseline") or {}
                ).get("curves"),
                importance=primary["feature_importance"],
                model_kind=primary["kind"],
            )

    except BaseException:
        shutil.rmtree(
            run_dir,
            ignore_errors=True,
        )
        raise

    selected = max(
        results,
        key=lambda role: (
            results[role]["validation"]["pr_auc"]
            if results[role]["validation"]["pr_auc"] is not None
            else -1.0
        ),
    )

    split_info = {
        **steps,
        "train_steps": [
            int(tr["step"].min()),
            int(tr["step"].max()),
        ],
        "validation_steps": [
            int(va["step"].min()),
            int(va["step"].max()),
        ],
        "test_steps": [
            int(te["step"].min()),
            int(te["step"].max()),
        ],
        "class_balance": {
            name: {
                "n": int(len(part)),
                "n_fraud": int(part[TARGET].sum()),
                "prevalence": float(part[TARGET].mean()),
            }
            for name, part in (
                ("train", tr),
                ("validation", va),
                ("test", te),
            )
        },
    }

    summary = {
        "run_id": run_id,
        "data_label": data_label,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "feature_set": feature_set,
        "n_features": len(contract.feature_names),
        "split": split_info,
        "selected_by_validation_pr_auc": selected,
        "search": search,
        "plots": plots_written,
        "models": {
            role: {
                key: value
                for key, value in result.items()
                if key != "test_scores"
            }
            for role, result in results.items()
        },
        "resources": {
            "total_seconds": round(
                time.perf_counter() - t_start,
                2,
            ),
            "peak_memory_mb": peak_memory_mb(),
            **environment_info(),
        },
        "mlflow": {
            "status": "disabled",
        },
    }

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (run_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=_jsonable,
        ),
        encoding="utf-8",
    )

    (run_dir / "report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )

    if use_mlflow:
        summary["mlflow"] = log_to_mlflow(
            summary,
            run_dir,
        )

        (run_dir / "summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                default=_jsonable,
            ),
            encoding="utf-8",
        )

        (run_dir / "report.md").write_text(
            render_report(summary),
            encoding="utf-8",
        )

    summary["run_dir"] = str(run_dir)

    summary["_test_scores"] = {
        role: result["test_scores"]
        for role, result in results.items()
    }

    return summary


def log_to_mlflow(
    summary: dict,
    run_dir: Path,
) -> dict:
    """Best effort MLflow logging. Training never fails because MLflow is missing."""
    try:
        import mlflow
    except ImportError:
        return {
            "status": "unavailable: mlflow not installed"
        }

    try:
        uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or (ROOT / "mlruns").resolve().as_uri()
        )

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("aiml-01-fraud")

        with mlflow.start_run(
            run_name=summary["run_id"]
        ):
            active_run = mlflow.active_run()
            info = (
                active_run.info
                if active_run is not None
                else None
            )

            mlflow_result = {
                "status": "logged",
                "experiment": "aiml-01-fraud",
                "tracking_uri": uri,
                "run_id": getattr(info, "run_id", None),
                "artifact_uri": getattr(info, "artifact_uri", None),
            }

            summary["mlflow"] = mlflow_result

            (run_dir / "summary.json").write_text(
                json.dumps(
                    summary,
                    indent=2,
                    default=_jsonable,
                ),
                encoding="utf-8",
            )

            (run_dir / "report.md").write_text(
                render_report(summary),
                encoding="utf-8",
            )

            if hasattr(mlflow, "set_tags"):
                mlflow.set_tags({
                    "project": "aiml-01-fraud-detection",
                    "data_label": summary["data_label"],
                    "feature_set": summary["feature_set"],
                })

            split = summary["split"]

            mlflow.log_params({
                "seed": summary["seed"],
                "dataset_sha256": (
                    summary["dataset_sha256"] or "none"
                ),
                "feature_set": summary["feature_set"],
                "n_features": summary["n_features"],
                "train_steps": str(split["train_steps"]),
                "validation_steps": str(split["validation_steps"]),
                "test_steps": str(split["test_steps"]),
                **{
                    f"primary_{key}": str(value)
                    for key, value
                    in summary["models"]["primary"]["params"].items()
                },
                **{
                    f"{role}_kind": model["kind"]
                    for role, model in summary["models"].items()
                },
            })

            for role, model in summary["models"].items():
                metrics = {
                    **_flat(
                        f"{role}_val_",
                        model["validation"],
                    ),
                    **_flat(
                        f"{role}_test_",
                        model["test"],
                    ),
                    f"{role}_fit_seconds": model["fit_seconds"],
                }

                if metrics:
                    mlflow.log_metrics(metrics)

            mlflow.log_artifacts(str(run_dir))

        return mlflow_result

    except Exception as exc:
        return {
            "status": f"failed: {type(exc).__name__}"
        }


def _fmt(value, pct=False):
    if value is None:
        return "n/a"

    return (
        f"{value:.4%}"
        if pct
        else f"{value:.4f}"
    )


def render_report(summary: dict) -> str:
    """Generate Markdown training report."""
    banner = (
        "> **SYNTHETIC DATA - these numbers are NOT real-world "
        "or PaySim results.**\n\n"
        if summary["data_label"].upper().startswith("SYNTH")
        else ""
    )

    resources = summary.get("resources", {})

    output = [
        f"# Training report: {summary['run_id']}",
        "",
        banner + f"- Data: {summary['data_label']}",
        f"- Dataset SHA-256: `{summary['dataset_sha256']}`",
        f"- Seed: {summary['seed']}",
        (
            f"- Feature set: **{summary.get('feature_set', 'base')}** "
            f"({summary.get('n_features', '?')} features)"
        ),
        (
            f"- Split (by step): train {summary['split']['train_steps']}, "
            f"validation {summary['split']['validation_steps']}, "
            f"test {summary['split']['test_steps']}"
        ),
        "- Class balance: " + "; ".join(
            f"{name}: {values['n_fraud']:,} fraud "
            f"of {values['n']:,} ({values['prevalence']:.4%})"
            for name, values
            in summary["split"]["class_balance"].items()
        ),
        (
            "- Model selected by VALIDATION PR-AUC: "
            f"**{summary['selected_by_validation_pr_auc']}**"
        ),
        (
            f"- Run time: {resources.get('total_seconds', '?')} s; "
            f"peak memory: {resources.get('peak_memory_mb', 'n/a')} MB; "
            f"CPUs: {resources.get('cpu_count', '?')}; "
            f"Python {resources.get('python', '?')}"
        ),
        f"- MLflow: {summary['mlflow']['status']}",
        "",
        (
            "Scores are ranking scores from class-weighted models, "
            "not calibrated probabilities."
        ),
        "",
    ]

    if summary.get("feature_set") == "all_balance":
        output += [
            "> **DIAGNOSTIC RUN:** this feature set includes "
            "post-transaction balances (newbalanceOrig / newbalanceDest "
            "and their error terms). They are not known when a transaction "
            "is scored and leak the label in PaySim. Do not present these "
            "results as deployable performance.",
            "",
        ]

    if summary.get("search"):
        output += [
            "## Hyperparameter search "
            "(validation PR-AUC only; test split not used)",
            "",
            "| params | validation PR-AUC | fit seconds |",
            "|---|---|---|",
            *[
                f"| {trial['params']} | "
                f"{_fmt(trial['val_pr_auc'])} | "
                f"{trial['fit_seconds']} |"
                for trial in summary["search"]["trials"]
            ],
            "",
            f"Chosen: `{summary['search']['chosen']}`",
            "",
        ]

    for role, model in summary["models"].items():
        test_metrics = model["test"]
        val_metrics = model["validation"]
        threshold = model["threshold"]

        output += [
            f"## {role}: {model['kind']}",
            "",
            (
                f"Threshold {threshold['threshold']:.6f} chosen on "
                f"validation ({threshold['policy']}"
                f"{', FALLBACK to max F1' if threshold['fallback_to_max_f1'] else ''}). "
                f"Fit time {model['fit_seconds']} s. "
                f"Reload check: "
                f"{'passed' if model['reload_check']['passed'] else 'FAILED'} "
                f"(max |diff| {model['reload_check']['max_abs_diff']:.2e} "
                f"on {model['reload_check']['rows']} held-out rows)."
            ),
            "",
            (
                "| set | PR-AUC | ROC-AUC | precision | recall | "
                "F1 | TP | FP | FN | TN |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for name, metrics in (
            ("validation", val_metrics),
            ("test", test_metrics),
        ):
            output.append(
                f"| {name} | {_fmt(metrics['pr_auc'])} | "
                f"{_fmt(metrics['roc_auc'])} | "
                f"{_fmt(metrics['precision'])} | "
                f"{_fmt(metrics['recall'])} | "
                f"{_fmt(metrics['f1'])} | "
                f"{metrics['tp']} | {metrics['fp']} | "
                f"{metrics['fn']} | {metrics['tn']} |"
            )

        cr = model.get("classification_report")

        if cr:
            output += [
                "",
                "Classification report "
                "(held-out test, at the validation-chosen threshold):",
                "",
                "| class | precision | recall | F1 | support |",
                "|---|---|---|---|---|",
            ]

            output += [
                f"| {key} | "
                f"{_fmt(cr[key]['precision'])} | "
                f"{_fmt(cr[key]['recall'])} | "
                f"{_fmt(cr[key]['f1-score'])} | "
                f"{int(cr[key]['support']):,} |"
                for key in (
                    "not_fraud",
                    "fraud",
                    "macro avg",
                    "weighted avg",
                )
            ]

        importance = model.get("feature_importance")

        if importance:
            output += [
                "",
                f"Top features ({importance['type']}): "
                + ", ".join(
                    f"{name} ({value:.3f})"
                    for name, value in zip(
                        importance["features"][:8],
                        importance["values"][:8],
                    )
                ),
            ]

        output += [
            "",
            "Test threshold analysis:",
            "",
            "| threshold | precision | recall | F1 | flagged rate |",
            "|---|---|---|---|---|",
        ]

        output += [
            f"| {row['threshold']:.2f} | "
            f"{_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | "
            f"{_fmt(row['f1'])} | "
            f"{_fmt(row['flagged_rate'], True)} |"
            for row in model["test_threshold_table"]
        ]

        output.append("")

    if summary.get("plots"):
        output += [
            "## Plots (plots/ folder; titles carry the data label)",
            "",
            *[
                f"- `{filename}`"
                for filename in summary["plots"]
            ],
            "",
        ]

    return "\n".join(output) + "\n"


HYPER_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "n_jobs",
    "early_stopping_rounds",
)


def params_from_args(kind: str, a) -> dict | None:
    """Map CLI hyperparameter flags to the chosen model kind."""
    given = {
        key: getattr(a, key)
        for key in HYPER_KEYS
        if getattr(a, key, None) is not None
    }

    if not given:
        return None

    if kind == "xgboost":
        return given

    if kind == "hist_gbm":
        unsupported = set(given) - {
            "n_estimators",
            "max_depth",
            "learning_rate",
        }

        if unsupported:
            raise ValueError(
                f"Options {sorted(unsupported)} "
                "are not supported for --model hist_gbm"
            )

        return {
            "max_iter" if key == "n_estimators" else key: value
            for key, value in given.items()
        }

    raise ValueError(
        "Hyperparameter options do not apply to --model logreg"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train and evaluate the fraud model."
    )

    ap.add_argument(
        "--data",
        required=True,
        help="Path to the PaySim CSV",
    )

    ap.add_argument(
        "--model",
        default="xgboost",
        choices=MODEL_KINDS,
    )

    ap.add_argument(
        "--feature-set",
        default="base",
        choices=FEATURE_SETS,
        help=(
            "base: no balances | pre_balance: adds balances known "
            "BEFORE the transaction | all_balance: DIAGNOSTIC, "
            "includes post-transaction balances (leakage)"
        ),
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)

    ap.add_argument(
        "--nrows",
        type=int,
        default=None,
        help=(
            "Development only: first N rows in file order "
            "(PaySim is step-sorted, so this is the earliest period)"
        ),
    )

    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--colsample-bytree", type=float, default=None)

    ap.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="CPU threads for xgboost (default: all CPUs)",
    )

    ap.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--search",
        action="store_true",
        help=(
            "Small grid (max_depth x learning_rate) "
            "scored on VALIDATION PR-AUC only"
        ),
    )

    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--run-id", default=None)

    ap.add_argument(
        "--data-label",
        default=None,
        help="Provenance label for reports.",
    )

    ap.add_argument("--no-mlflow", action="store_true")

    a = ap.parse_args(argv)

    try:
        path = Path(a.data)

        df = load_dataset(
            path,
            a.nrows,
            a.feature_set,
        )

        s = run_training(
            df,
            kind=a.model,
            seed=a.seed,
            min_precision=a.min_precision,
            params=params_from_args(a.model, a),
            out_root=Path(a.out_dir),
            run_id=a.run_id,
            data_label=(
                a.data_label
                or (
                    f"user-supplied file {path.name} "
                    "(provenance unverified)"
                    + (
                        f", first {a.nrows} rows"
                        if a.nrows
                        else ""
                    )
                )
            ),
            dataset_sha256=sha256_file(path),
            train_frac=a.train_frac,
            val_frac=a.val_frac,
            use_mlflow=not a.no_mlflow,
            feature_set=a.feature_set,
            grid=DEFAULT_GRID if a.search else None,
            make_plots=not a.no_plots,
        )

    except (
        DatasetError,
        SchemaError,
        TrainingError,
        MissingDependencyError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Run {s['run_id']} written to {s['run_dir']}"
    )

    print(
        (
            Path(s["run_dir"]) / "report.md"
        ).read_text(encoding="utf-8")
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())