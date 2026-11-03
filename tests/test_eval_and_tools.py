"""Evaluation outputs, search, reload verification, splits/compare CLIs, EDA analysis, field discovery. SYNTHETIC data only."""
import inspect
import json
import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises, skip  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

import ml.train as train_mod  # noqa: E402
from ml.compare_runs import load_rows, main as compare_main  # noqa: E402
from ml.eda.analyze import amount_by_class, balance_analysis, fraud_per_step, main as eda_main  # noqa: E402
from ml.pipeline.evaluate import classification_report_dict, curve_points  # noqa: E402
from ml.pipeline.model_io import load_artifact  # noqa: E402
from ml.pipeline.models import ScoreModel  # noqa: E402
from ml.pipeline.resources import environment_info, peak_memory_mb  # noqa: E402
from ml.splits import describe_splits, main as splits_main  # noqa: E402
from ml.train import TrainingError, main as train_main, params_from_args, run_training  # noqa: E402
from streaming.outcomes import OutcomeRecorder  # noqa: E402
from streaming.replay import ApiSink, load_rows as replay_rows  # noqa: E402


def fit(tmp_path, run_id="r", **kw):
    kw.setdefault("params", {"max_iter": 30})
    return run_training(make_synthetic(), kind="hist_gbm", out_root=tmp_path, run_id=run_id,
                        data_label="SYNTHETIC test fixture", use_mlflow=False, **kw)


# ---------------- metrics helpers ----------------
def test_classification_report_values_and_missing_class():
    r = classification_report_dict([1, 1, 0, 0, 0, 0], [0.9, 0.4, 0.8, 0.2, 0.1, 0.05], 0.5)
    assert r["fraud"]["support"] == 2 and r["not_fraud"]["support"] == 4
    assert r["fraud"]["precision"] == 0.5 and r["fraud"]["recall"] == 0.5
    only_neg = classification_report_dict([0, 0, 0], [0.1, 0.2, 0.9], 0.5)  # no fraud rows: zeros, not an exception
    assert only_neg["fraud"]["support"] == 0 and only_neg["fraud"]["recall"] == 0.0


def test_curve_points_downsample_and_undefined_cases():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    c = curve_points(y, rng.random(5000), max_points=100)
    assert len(c["pr"]["precision"]) <= 100 and len(c["roc"]["fpr"]) <= 100 and abs(c["prevalence"] - y.mean()) < 1e-12
    assert c["roc"]["fpr"][0] == 0.0 and c["roc"]["tpr"][-1] == 1.0
    assert curve_points([0, 0, 0], [0.1, 0.2, 0.3]) is None and curve_points([1, 1], [0.1, 0.2]) is None


def test_feature_importance_by_model_kind():
    X = make_synthetic(10, 10)
    from ml.pipeline.features import FeatureContract, build_features
    c = FeatureContract.fit(X)
    F, y = build_features(X, c), X["isFraud"]
    lr = ScoreModel("logreg", seed=1).fit(F, y)
    imp = lr.feature_importance(c.feature_names, top=5)
    assert imp["type"] == "abs_standardised_coefficient" and len(imp["features"]) == 5 and imp["values"] == sorted(imp["values"], reverse=True)
    assert ScoreModel("hist_gbm", params={"max_iter": 10}, seed=1).fit(F, y).feature_importance(c.feature_names) is None
    stub = ScoreModel("xgboost")
    stub.est = types.SimpleNamespace(feature_importances_=np.array([0.1, 0.7, 0.2]))
    s = stub.feature_importance(["a", "b", "c"])
    assert s["type"] == "gain" and s["features"] == ["b", "c", "a"]


def test_resources_helpers():
    m = peak_memory_mb()
    assert m is None or m > 0
    assert {"python", "platform", "cpu_count"} <= set(environment_info())


# ---------------- training outputs ----------------
def test_training_writes_report_curves_importance_and_timing(tmp_path):
    s = fit(tmp_path, make_plots=False)
    run = Path(s["run_dir"])
    prim = s["models"]["primary"]
    assert prim["classification_report"]["fraud"]["support"] == prim["test"]["n_fraud"]
    assert prim["curves"]["prevalence"] == prim["test"]["prevalence"] and prim["fit_seconds"] >= 0
    assert s["resources"]["total_seconds"] >= prim["fit_seconds"] and s["resources"]["cpu_count"] >= 1
    meta = load_artifact(run / "primary").metadata
    assert meta["training"]["fit_seconds"] == prim["fit_seconds"] and "test_curves" in meta["metrics"]
    txt = (run / "report.md").read_text()
    assert "Classification report" in txt and "Reload check: passed" in txt and "Run time:" in txt
    assert json.loads((run / "summary.json").read_text())["models"]["primary"]["reload_check"]["passed"] is True


def test_plots_are_written_and_labelled_files_are_real_pngs(tmp_path):
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        skip("matplotlib not installed")
    s = fit(tmp_path, run_id="pl", params=None, kind_override=None) if False else fit(tmp_path, run_id="pl")
    assert set(s["plots"]) == {"confusion_matrix.png", "precision_recall_curve.png", "roc_curve.png"}  # hist_gbm has no importance
    for f in s["plots"]:
        data = (Path(s["run_dir"]) / "plots" / f).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 2000
    lr = run_training(make_synthetic(), kind="logreg", out_root=tmp_path, run_id="pl2", data_label="SYNTHETIC test fixture", use_mlflow=False)
    assert "feature_importance.png" in lr["plots"]


def test_reload_check_failure_aborts_and_cleans_up(tmp_path):
    original = train_mod._verify_reload
    train_mod._verify_reload = lambda *a, **k: {"rows": 1, "max_abs_diff": 1.0, "tolerance": 1e-6, "feature_schema_match": True, "passed": False}
    try:
        e = assert_raises(TrainingError, fit, tmp_path, "bad")
    finally:
        train_mod._verify_reload = original
    assert "reload check" in str(e) and not (tmp_path / "bad").exists()


def test_reloaded_model_matches_in_memory_model(tmp_path):
    s = fit(tmp_path)
    lm = load_artifact(Path(s["run_dir"]) / "primary")
    assert s["models"]["primary"]["reload_check"]["max_abs_diff"] <= 1e-6
    df = make_synthetic()
    te = df["step"] > s["split"]["val_max_step"]
    assert np.allclose(lm.predict_score(df)[te.to_numpy()], s["_test_scores"]["primary"])


def test_inference_needs_no_training_only_packages(tmp_path):
    s = fit(tmp_path)
    saved = {k: sys.modules.get(k, "absent") for k in ("mlflow", "matplotlib", "matplotlib.pyplot")}
    for k in saved:
        sys.modules[k] = None  # importing any of them now raises ImportError
    try:
        lm = load_artifact(Path(s["run_dir"]) / "primary")
        out = lm.predict_score(make_synthetic(3, 5))
    finally:
        for k, v in saved.items():
            if v == "absent":
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert len(out) == 15


# ---------------- hyperparameters and search ----------------
def test_search_uses_validation_only_and_picks_the_best_trial(tmp_path):
    s = fit(tmp_path, grid={"max_depth": [2, 5], "learning_rate": [0.05, 0.3]}, make_plots=False)
    tr = s["search"]["trials"]
    assert len(tr) == 4 and s["search"]["test_data_used"] is False and s["search"]["selection_metric"] == "validation PR-AUC"
    best = max(tr, key=lambda t: t["val_pr_auc"])
    assert s["search"]["chosen"] == best["params"]
    used = s["models"]["primary"]["params"]
    assert used["max_depth"] == best["params"]["max_depth"] and used["learning_rate"] == best["params"]["learning_rate"]
    assert "te" not in inspect.signature(train_mod._search).parameters  # the test split cannot be passed in
    assert "Hyperparameter search" in (Path(s["run_dir"]) / "report.md").read_text()


def test_search_not_allowed_for_baseline(tmp_path):
    assert_raises(TrainingError, run_training, make_synthetic(), kind="logreg", out_root=tmp_path, grid={"max_depth": [2]}, use_mlflow=False)


def test_params_from_args_mapping():
    ns = lambda **k: types.SimpleNamespace(**{x: k.get(x) for x in train_mod.HYPER_KEYS})  # noqa: E731
    assert params_from_args("xgboost", ns(n_estimators=10, max_depth=3, n_jobs=2)) == {"n_estimators": 10, "max_depth": 3, "n_jobs": 2}
    assert params_from_args("hist_gbm", ns(n_estimators=10, learning_rate=0.2)) == {"max_iter": 10, "learning_rate": 0.2}
    assert params_from_args("xgboost", ns()) is None
    assert_raises(ValueError, params_from_args, "hist_gbm", ns(subsample=0.5))
    assert_raises(ValueError, params_from_args, "logreg", ns(max_depth=3))


def test_train_cli_with_feature_set_hyperparameters_and_errors(tmp_path):
    f = tmp_path / "s.csv"
    make_synthetic().to_csv(f, index=False)
    common = ["--data", str(f), "--no-mlflow", "--no-plots", "--data-label", "SYNTHETIC test CSV", "--out-dir", str(tmp_path / "o")]
    assert train_main(["--model", "hist_gbm", "--feature-set", "pre_balance", "--n-estimators", "20", "--max-depth", "3",
                       "--run-id", "c1", *common]) == 0
    s = json.loads((tmp_path / "o" / "c1" / "summary.json").read_text())
    assert s["feature_set"] == "pre_balance" and s["models"]["primary"]["params"]["max_iter"] == 20 and s["models"]["primary"]["params"]["max_depth"] == 3
    assert train_main(["--model", "hist_gbm", "--subsample", "0.5", "--run-id", "c2", *common]) == 2
    assert train_main(["--model", "hist_gbm", "--search", "--n-estimators", "15", "--run-id", "c3", *common]) == 0
    assert len(json.loads((tmp_path / "o" / "c3" / "summary.json").read_text())["search"]["trials"]) == 4


# ---------------- splits and compare ----------------
def test_describe_splits_and_cli(tmp_path):
    d = describe_splits(make_synthetic())
    assert d["all_splits_usable"] and d["splits"]["train"]["steps"][1] < d["splits"]["validation"]["steps"][0] <= d["splits"]["validation"]["steps"][1] < d["splits"]["test"]["steps"][0]
    assert sum(s["rows"] for s in d["splits"].values()) == 60 * 40
    f = tmp_path / "s.csv"
    make_synthetic().to_csv(f, index=False)
    assert splits_main(["--data", str(f), "--json"]) == 0
    late = make_synthetic()
    late.loc[late["step"] > 36, "isFraud"] = 0
    g = tmp_path / "late.csv"
    late.to_csv(g, index=False)
    assert splits_main(["--data", str(g)]) == 1  # unusable split flagged via exit code
    assert describe_splits(late)["splits"]["test"]["usable"] is False
    assert splits_main(["--data", str(tmp_path / "missing.csv")]) == 2


def test_compare_runs(tmp_path):
    a, b = fit(tmp_path, "ra"), fit(tmp_path, "rb", feature_set="pre_balance")
    rows = load_rows([a["run_dir"], b["run_dir"]])
    assert {r["feature_set"] for r in rows} == {"base", "pre_balance"} and len(rows) == 4  # baseline + primary each
    assert compare_main([a["run_dir"], b["run_dir"]]) == 0
    assert compare_main([str(tmp_path / "nope")]) == 2 and compare_main([]) == 2


# ---------------- EDA analysis ----------------
def tiny_eda_df() -> pd.DataFrame:
    return pd.DataFrame(dict(
        step=[1, 1, 2, 2, 3], type=["TRANSFER", "TRANSFER", "TRANSFER", "PAYMENT", "CASH_OUT"],
        amount=[100.0, 50.0, 10.0, 5.0, 20.0], nameOrig=list("ABCDE"), nameDest=["X", "Y", "Z", "M1", "W"],
        oldbalanceOrg=[100.0, 50.0, 500.0, 100.0, 20.0], newbalanceOrig=[0.0, 0.0, 490.0, 95.0, 0.0],
        oldbalanceDest=[0.0, 0.0, 0.0, 0.0, 0.0], newbalanceDest=[0.0, 0.0, 10.0, 0.0, 0.0],
        isFraud=[1, 1, 0, 0, 0], isFlaggedFraud=[0] * 5))


def test_balance_analysis_known_values():
    b = balance_analysis(tiny_eda_df())
    f, n = b["by_class"]["fraud"], b["by_class"]["non_fraud"]
    assert f["rows"] == 2 and n["rows"] == 3
    assert f["share_orig_emptied"] == 1.0 and abs(n["share_orig_emptied"] - 1 / 3) < 1e-12  # the CASH_OUT non-fraud also empties
    assert f["share_newbalanceOrig_zero"] == 1.0 and f["share_dest_error_nonzero"] == 1.0   # fraud: dest never credited
    assert n["share_orig_error_nonzero"] == 0.0 and n["share_dest_is_merchant"] == 1 / 3
    d = b["single_rule_diagnostic"]
    assert d["types_used"] == ["TRANSFER"] and (d["tp"], d["fp"], d["fn"]) == (2, 0, 0) and d["precision"] == 1.0 and d["recall"] == 1.0
    assert_raises(ValueError, balance_analysis, tiny_eda_df().drop(columns=["newbalanceDest"]))


def test_amount_and_step_summaries():
    df = tiny_eda_df()
    a = amount_by_class(df)
    assert a["fraud"]["count"] == 2 and a["fraud"]["max"] == 100.0 and a["non_fraud"]["count"] == 3
    s = fraud_per_step(df)
    assert s["steps"] == 3 and s["fraud_per_step"]["max"] == 2 and s["steps_with_zero_fraud"] == 2 and s["series"]["fraud"] == [2, 0, 0]


def test_eda_cli_generates_report_json_and_plots_from_the_given_file(tmp_path):
    f = tmp_path / "SYNTH_for_test.csv"
    make_synthetic(20, 30).to_csv(f, index=False)
    out = tmp_path / "eda"
    rep = tmp_path / "eda_report.md"
    assert eda_main(["--data", str(f), "--out-dir", str(out), "--report", str(rep)]) == 0
    txt = rep.read_text()
    assert "SYNTH_for_test.csv" in txt and "SHA-256" in txt and "simulation" in txt and "Leakage / generator-rule diagnostic" in txt
    assert json.loads((out / "balance_analysis.json").read_text())["balance"]["single_rule_diagnostic"]["rule"]
    try:
        import matplotlib  # noqa: F401
        assert {p.name for p in (out / "plots").iterdir()} == {"fraud_rate_by_type.png", "amount_by_class.png", "fraud_per_step.png", "class_balance.png"}
    except ImportError:
        pass


def test_eda_cli_errors_cleanly_without_a_dataset(tmp_path):
    assert eda_main(["--data", str(tmp_path / "missing.csv"), "--out-dir", str(tmp_path), "--report", str(tmp_path / "r.md")]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert eda_main(["--data", str(bad), "--out-dir", str(tmp_path), "--report", str(tmp_path / "r.md")]) in (0, 2)


# ---------------- replay field discovery ----------------
class _Api:
    def __init__(self, model_body, status=200):
        outer = self
        self.calls = []

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                raw = json.dumps(model_body).encode()
                self.send_response(status); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                outer.calls.append((self.path, self.rfile.read(n)))
                raw = b'{"flagged": false}'
                self.send_response(200); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

            def log_message(self, *a):
                pass
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/api/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


def test_replay_discovers_required_fields_and_sends_exactly_those(tmp_path):
    f = tmp_path / "s.csv"
    make_synthetic(3, 4).to_csv(f, index=False)
    fields = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "oldbalanceDest"]
    api = _Api({"state": "ready", "model": {"input_fields": fields}})
    try:
        sink = ApiSink(api.url, OutcomeRecorder(None, "api"))
        got = sink.discover_fields()
        rows = replay_rows(f, None, got)
        sink.send_one(rows[0], 0)
    finally:
        api.close()
    assert list(got) == fields and set(rows[0]) == set(fields)
    assert set(json.loads(api.calls[0][1])) == set(fields)  # no post-transaction columns, no label, no flag


def test_replay_discovery_falls_back_to_base_safely(tmp_path):
    for body in ({"state": "unavailable", "model": None}, {"model": {"input_fields": ["step", "evil_field"]}}, {"model": {"input_fields": ["oldbalanceOrg"]}}):
        api = _Api(body)
        try:
            assert ApiSink(api.url, OutcomeRecorder(None, "x")).discover_fields() == ("step", "type", "amount", "nameOrig", "nameDest")
        finally:
            api.close()
    api = _Api({}, status=500)
    try:
        assert len(ApiSink(api.url, OutcomeRecorder(None, "x")).discover_fields()) == 5
    finally:
        api.close()
