"""All data here is SYNTHETIC (tests/synthetic.py). Real-data training is NOT covered."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises, skip  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

from ml.pipeline import models as models_mod  # noqa: E402
from ml.pipeline.evaluate import (compute_metrics, select_threshold,  # noqa: E402
                                  threshold_table)
from ml.pipeline.features import FORBIDDEN_FEATURE_COLUMNS  # noqa: E402
from ml.pipeline.model_io import ArtifactError, load_artifact  # noqa: E402
from ml.pipeline.models import MissingDependencyError, ScoreModel  # noqa: E402
from ml.pipeline.schema import SchemaError  # noqa: E402
from ml.train import DatasetError, TrainingError, load_dataset, main, run_training  # noqa: E402

FAST = {"max_iter": 40}


def train(tmp_path, kind="hist_gbm", seed=1, df=None, run_id="r1", **kw):
    df = make_synthetic() if df is None else df
    return run_training(df, kind=kind, seed=seed, params=FAST if kind == "hist_gbm" else None,
                        out_root=tmp_path, run_id=run_id, data_label="SYNTHETIC test fixture",
                        use_mlflow=False, **kw)


def has_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# ---------- metrics / thresholds ----------
def test_metrics_known_values():
    y = [1, 1, 0, 0, 0, 0]
    s = [0.9, 0.4, 0.8, 0.2, 0.1, 0.05]
    m = compute_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert 0 < m["pr_auc"] <= 1 and 0 < m["roc_auc"] <= 1 and m["n_fraud"] == 2


def test_metrics_undefined_cases_are_none_not_invented():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["precision"] == 0.0


def test_select_threshold_respects_min_precision_and_fallback():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    s = [0.95, 0.9, 0.3, 0.8, 0.2, 0.1, 0.05, 0.01]
    r = select_threshold(y, s, 0.6)
    assert r["val_precision"] >= 0.6 and not r["fallback_to_max_f1"]
    r2 = select_threshold([1, 0, 0, 1], [0.1, 0.9, 0.8, 0.2], 0.99)
    assert r2["fallback_to_max_f1"]
    assert_raises(ValueError, select_threshold, [0, 0], [0.1, 0.2])
    assert_raises(ValueError, select_threshold, [1, 0], [0.9, 0.1], 0)


def test_threshold_table_shape():
    t = threshold_table([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.4])
    assert len(t) == 19 and {"threshold", "precision", "recall", "f1", "tp"} <= set(t[0])


# ---------- training end to end (scikit-learn backends) ----------
def test_training_writes_artifacts_and_metrics(tmp_path):
    s = train(tmp_path)
    run = Path(s["run_dir"])
    assert (run / "summary.json").is_file() and (run / "report.md").is_file()
    for role in ("baseline", "primary"):
        assert (run / role / "metadata.json").is_file() and (run / role / "model.joblib").is_file()
    t = s["models"]["primary"]["test"]
    assert t["pr_auc"] > t["prevalence"]  # weak sanity: better than chance on planted signal
    assert "SYNTHETIC" in (run / "report.md").read_text(encoding="utf-8")
    assert s["selected_by_validation_pr_auc"] in ("baseline", "primary")


def test_training_is_reproducible_for_same_seed(tmp_path):
    a, b = train(tmp_path, run_id="a"), train(tmp_path, run_id="b")
    assert np.array_equal(a["_test_scores"]["primary"], b["_test_scores"]["primary"])
    assert a["models"]["primary"]["test"] == b["models"]["primary"]["test"]


def test_time_split_recorded_and_ordered(tmp_path):
    sp = train(tmp_path)["split"]
    assert sp["train_steps"][1] < sp["validation_steps"][0]
    assert sp["validation_steps"][1] < sp["test_steps"][0]


def test_threshold_fixed_on_validation_not_test(tmp_path):
    s = train(tmp_path)
    m = s["models"]["primary"]
    assert m["threshold"]["threshold"] == m["validation"]["threshold"] == m["test"]["threshold"]


# ---------- save / load ----------
def test_save_load_roundtrip_gives_identical_scores(tmp_path):
    s = train(tmp_path)
    lm = load_artifact(Path(s["run_dir"]) / "primary")
    df = make_synthetic()
    te_mask = df["step"] > s["split"]["val_max_step"]
    scores = lm.predict_score(df)[te_mask.to_numpy()]
    # test rows are scored the same way as during training (features built over the full frame)
    assert np.allclose(scores, s["_test_scores"]["primary"])
    assert set(lm.predict_label(df)) <= {0, 1}


def test_load_missing_artifact_has_useful_error(tmp_path):
    e = assert_raises(ArtifactError, load_artifact, tmp_path / "nope")
    assert "Train a model first" in str(e)


def test_tampered_model_file_is_rejected(tmp_path):
    d = Path(train(tmp_path)["run_dir"]) / "primary"
    with (d / "model.joblib").open("ab") as f:
        f.write(b"x")
    assert "SHA-256" in str(assert_raises(ArtifactError, load_artifact, d))


def test_corrupt_and_incompatible_metadata_rejected(tmp_path):
    d = Path(train(tmp_path)["run_dir"]) / "primary"
    meta_p = d / "metadata.json"
    good = json.loads(meta_p.read_text())
    meta_p.write_text("{not json")
    assert_raises(ArtifactError, load_artifact, d)
    bad = json.loads(json.dumps(good))
    bad["library_versions"]["sklearn"] = "0.1.0"
    meta_p.write_text(json.dumps(bad))
    assert "version mismatch" in str(assert_raises(ArtifactError, load_artifact, d))
    bad = json.loads(json.dumps(good))
    bad["feature_contract"]["version"] = "999"
    meta_p.write_text(json.dumps(bad))
    assert "feature contract" in str(assert_raises(ArtifactError, load_artifact, d)).lower()
    bad = json.loads(json.dumps(good))
    del bad["threshold"]
    meta_p.write_text(json.dumps(bad))
    assert_raises(ArtifactError, load_artifact, d)


def test_metadata_cannot_redirect_to_other_paths(tmp_path):
    d = Path(train(tmp_path)["run_dir"]) / "primary"
    meta = json.loads((d / "metadata.json").read_text())
    meta["model_file"] = "../baseline/model.joblib"  # traversal is reduced to the bare file name
    (d / "metadata.json").write_text(json.dumps(meta))
    lm = load_artifact(d)  # loads this directory's own model.joblib, not ../baseline
    assert lm.metadata["model_kind"] == "hist_gbm"
    meta["model_file"] = "../baseline/other.joblib"
    (d / "metadata.json").write_text(json.dumps(meta))
    assert_raises(ArtifactError, load_artifact, d)


# ---------- leakage ----------
def test_artifact_features_exclude_leakage_columns(tmp_path):
    s = train(tmp_path)
    for role in ("baseline", "primary"):
        meta = json.loads((Path(s["run_dir"]) / role / "metadata.json").read_text())
        names = set(meta["feature_contract"]["feature_names"])
        assert not names & set(FORBIDDEN_FEATURE_COLUMNS)


def test_serving_does_not_need_or_use_leaky_columns(tmp_path):
    lm = load_artifact(Path(train(tmp_path)["run_dir"]) / "primary")
    df = make_synthetic()
    base = lm.predict_score(df)
    no_leak_cols = df.drop(columns=list(FORBIDDEN_FEATURE_COLUMNS))
    assert np.array_equal(base, lm.predict_score(no_leak_cols))
    rng = np.random.default_rng(0)
    scrambled = df.copy()
    for c in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    assert np.array_equal(base, lm.predict_score(scrambled))


def test_future_rows_do_not_change_earlier_scores(tmp_path):
    lm = load_artifact(Path(train(tmp_path)["run_dir"]) / "primary")
    df = make_synthetic()
    cut = df["step"] <= 30
    full = lm.predict_score(df)[cut.to_numpy()]
    changed = df.copy()
    changed.loc[~cut, "amount"] = 1e9
    assert np.array_equal(full, lm.predict_score(changed)[cut.to_numpy()])
    assert np.array_equal(full, lm.predict_score(df[cut]))


def test_load_dataset_reads_only_needed_columns(tmp_path):
    f = tmp_path / "d.csv"
    make_synthetic(5, 10).to_csv(f, index=False)
    cols = set(load_dataset(f).columns)
    assert not cols & set(FORBIDDEN_FEATURE_COLUMNS) - {"isFraud"} and "isFraud" in cols


# ---------- invalid datasets ----------
def test_dataset_errors(tmp_path):
    assert "not found" in str(assert_raises(DatasetError, load_dataset, tmp_path / "x.csv"))
    assert_raises(DatasetError, load_dataset, tmp_path)
    t = tmp_path / "a.txt"
    t.write_text("a,b")
    assert_raises(DatasetError, load_dataset, t)
    e = tmp_path / "e.csv"
    e.write_text("")
    assert_raises(DatasetError, load_dataset, e)
    m = tmp_path / "m.csv"
    m.write_text("step,type\n1,PAYMENT\n")
    assert_raises(SchemaError, load_dataset, m)


def test_invalid_target_and_single_class_rejected(tmp_path):
    df = make_synthetic(10, 10)
    assert_raises(SchemaError, run_training, df.assign(isFraud=3), kind="logreg",
                  out_root=tmp_path, use_mlflow=False)
    assert_raises(SchemaError, run_training, df.assign(isFraud=0), kind="logreg",
                  out_root=tmp_path, use_mlflow=False)


def test_split_without_fraud_in_validation_or_test_is_rejected(tmp_path):
    df = make_synthetic()
    df.loc[df["step"] > 36, "isFraud"] = 0  # no fraud after step 36 -> val/test lack positives
    assert_raises(TrainingError, run_training, df, kind="logreg", out_root=tmp_path,
                  use_mlflow=False)


def test_existing_run_dir_is_not_overwritten(tmp_path):
    train(tmp_path, run_id="same")
    assert_raises(TrainingError, train, tmp_path, run_id="same")


def test_cli_returns_2_on_bad_input(tmp_path):
    assert main(["--data", str(tmp_path / "missing.csv"), "--model", "logreg",
                 "--out-dir", str(tmp_path)]) == 2


def test_cli_end_to_end_on_synthetic_csv(tmp_path):
    f = tmp_path / "synthetic.csv"
    make_synthetic().to_csv(f, index=False)
    code = main(["--data", str(f), "--model", "hist_gbm", "--no-mlflow", "--run-id", "cli",
                 "--data-label", "SYNTHETIC test CSV",
                 "--out-dir", str(tmp_path / "out")])
    assert code == 0 and (tmp_path / "out" / "cli" / "summary.json").is_file()
    s = json.loads((tmp_path / "out" / "cli" / "summary.json").read_text())
    assert len(s["dataset_sha256"]) == 64 and s["data_label"] == "SYNTHETIC test CSV"


# ---------- optional dependencies ----------
def test_mlflow_missing_does_not_break_training(tmp_path):
    saved = sys.modules.get("mlflow", "absent")
    sys.modules["mlflow"] = None  # forces ImportError even if installed
    try:
        s = run_training(make_synthetic(), kind="logreg", out_root=tmp_path, run_id="ml",
                         data_label="SYNTHETIC test fixture", use_mlflow=True)
    finally:
        if saved == "absent":
            del sys.modules["mlflow"]
        else:
            sys.modules["mlflow"] = saved
    assert s["mlflow"]["status"].startswith("unavailable")


def test_xgboost_missing_gives_clear_error_not_silent_fallback(tmp_path):
    if has_xgboost():
        skip("xgboost is installed; missing-dependency path not applicable here")
    e = assert_raises(MissingDependencyError, train, tmp_path, kind="xgboost")
    assert "hist_gbm" in str(e)
    assert not (tmp_path / "r1").exists()


def test_xgboost_wiring_with_stub_module():
    """STUB test: verifies OUR call arguments only. It does not exercise real XGBoost."""
    calls = {}

    class FakeClf:
        def __init__(self, **kw):
            calls["init"] = kw
            self.best_iteration = 7

        def fit(self, X, y, eval_set=None, verbose=None):
            calls["eval_set"] = eval_set

        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    fake = types.ModuleType("xgboost")
    fake.XGBClassifier, fake.__version__ = FakeClf, "0.0-stub"
    saved = sys.modules.get("xgboost")
    sys.modules["xgboost"] = fake
    try:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y = [0, 0, 0, 1]
        m = ScoreModel("xgboost", seed=5).fit(X, y, X, y)
    finally:
        if saved is None:
            del sys.modules["xgboost"]
        else:
            sys.modules["xgboost"] = saved
    kw = calls["init"]
    assert kw["scale_pos_weight"] == 3.0 and kw["random_state"] == 5
    assert kw["eval_metric"] == "aucpr" and kw["early_stopping_rounds"] == 30
    assert calls["eval_set"] is not None and m.fit_info["best_iteration"] == 7


def test_real_xgboost_end_to_end(tmp_path):
    if not has_xgboost():
        skip("BLOCKED: xgboost not installed in this environment")
    s = train(tmp_path, kind="xgboost", run_id="x")
    lm = load_artifact(Path(s["run_dir"]) / "primary")
    assert np.allclose(lm.predict_score(make_synthetic())[
        (make_synthetic()["step"] > s["split"]["val_max_step"]).to_numpy()],
        s["_test_scores"]["primary"])
    assert (Path(s["run_dir"]) / "primary" / "model.json").is_file()


def test_unknown_model_kind_rejected():
    assert_raises(ValueError, ScoreModel, "random_forest_of_doom")
    assert models_mod.MODEL_KINDS == ("xgboost", "hist_gbm", "logreg")
