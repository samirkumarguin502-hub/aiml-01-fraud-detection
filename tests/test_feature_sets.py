"""Feature sets, balance-aware API flows, DB migration. SYNTHETIC data only."""
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises  # noqa: E402
from synthetic import make_synthetic  # noqa: E402
from test_backend_core import GOOD, code_of, csv_bytes, make_service  # noqa: E402

from backend.app.db import Repository  # noqa: E402
from backend.app.errors import AppError  # noqa: E402
from ml.pipeline.features import (EXTRA_FEATURES, FORBIDDEN_FEATURE_COLUMNS, FeatureContract,  # noqa: E402
                                  build_features)
from ml.pipeline.model_io import load_artifact  # noqa: E402
from ml.pipeline.schema import (FEATURE_SET_INPUTS, SchemaError, feature_set_for_fields,  # noqa: E402
                                validate_schema)
from ml.train import DatasetError, load_dataset, run_training  # noqa: E402

_C: dict = {}
PRE = dict(oldbalanceOrg=5000.0, oldbalanceDest=0.0)


def trained(feature_set: str) -> Path:
    if feature_set not in _C:
        root = Path(tempfile.mkdtemp())
        run_training(make_synthetic(), kind="hist_gbm", params={"max_iter": 40}, out_root=root, run_id="fs",
                     data_label="SYNTHETIC test fixture", use_mlflow=False, feature_set=feature_set, make_plots=False)
        _C[feature_set] = root / "fs" / "primary"
    return _C[feature_set]


# ---------------- schema ----------------
def test_schema_requires_balance_columns_for_the_feature_set():
    df = make_synthetic(5, 5)
    assert validate_schema(df.drop(columns=["oldbalanceOrg"]))["rows"] == 25  # base does not need them
    for fs, drop in (("pre_balance", "oldbalanceDest"), ("all_balance", "newbalanceOrig")):
        assert "Missing required columns" in str(assert_raises(SchemaError, validate_schema, df.drop(columns=[drop]), feature_set=fs))
    assert_raises(SchemaError, validate_schema, df.assign(oldbalanceOrg=-1.0), feature_set="pre_balance")
    assert_raises(SchemaError, validate_schema, df, feature_set="nonsense")
    assert feature_set_for_fields(FEATURE_SET_INPUTS["pre_balance"]) == "pre_balance"
    assert feature_set_for_fields(FEATURE_SET_INPUTS["all_balance"]) == "all_balance"
    assert feature_set_for_fields(FEATURE_SET_INPUTS["base"]) == "base"


# ---------------- features ----------------
def test_feature_sets_add_exactly_the_documented_features():
    df = make_synthetic(5, 5)
    base = FeatureContract.fit(df, "base")
    for fs in ("pre_balance", "all_balance"):
        c = FeatureContract.fit(df, fs)
        assert c.feature_names[:len(base.feature_names)] == base.feature_names  # base order never changes
        assert c.feature_names[len(base.feature_names):] == EXTRA_FEATURES[fs]
        X = build_features(df, c)
        assert list(X.columns) == c.feature_names and not X.isna().any().any()


def test_isflaggedfraud_and_target_never_become_features_in_any_set():
    df = make_synthetic(5, 5)
    for fs in FEATURE_SET_INPUTS:
        c = FeatureContract.fit(df, fs)
        assert not set(c.feature_names) & {"isFraud", "isFlaggedFraud"}


def test_pre_balance_ignores_post_transaction_columns_and_labels():
    df = make_synthetic(20, 10)
    c = FeatureContract.fit(df, "pre_balance")
    a = build_features(df, c)
    b = build_features(df.assign(newbalanceOrig=123.0, newbalanceDest=9.0, isFlaggedFraud=1, isFraud=1 - df["isFraud"]), c)
    pd.testing.assert_frame_equal(a, b)
    assert not {"newbalanceOrig", "newbalanceDest"} & set(c.feature_names)
    leaky = FeatureContract.fit(df, "all_balance")  # diagnostic set DOES use post columns (documented)
    assert not build_features(df, leaky).equals(build_features(df.assign(newbalanceOrig=123.0), leaky))


def test_pre_balance_feature_values():
    df = pd.DataFrame(dict(step=[1, 2, 3, 4], type=["TRANSFER"] * 4, amount=[100.0, 50.0, 10.0, 7.0],
                           nameOrig=["A", "B", "C", "D"], nameDest=["X", "Y", "Z", "W"],
                           oldbalanceOrg=[100.0, 500.0, 0.0, 6.0], oldbalanceDest=[0.0, 20.0, 0.0, 1.0],
                           isFraud=[1, 0, 0, 0]))
    X = build_features(df, FeatureContract.fit(df, "pre_balance"))
    assert X["orig_emptied"].tolist() == [1, 0, 0, 1]            # amount takes whole balance; zero balance is not "emptied"
    assert X["orig_balance_zero"].tolist() == [0, 0, 1, 0]
    assert X["dest_balance_zero"].tolist() == [1, 0, 1, 0]
    assert np.allclose(X["amount_to_orig_balance"], [1.0, 0.1, 0.0, 7 / 6])


def test_history_stays_causal_with_balance_features():
    df = make_synthetic(30, 10)
    c = FeatureContract.fit(df, "pre_balance")
    base = build_features(df, c)
    changed = df.copy()
    changed.loc[changed["step"] >= 20, ["amount", "oldbalanceOrg"]] = 1e9
    after = build_features(changed, c)
    early = df["step"] < 20
    pd.testing.assert_frame_equal(base.loc[early, ["orig_prev_txn_count", "orig_prev_amount_mean"]],
                                  after.loc[early, ["orig_prev_txn_count", "orig_prev_amount_mean"]])


def test_contract_roundtrip_backward_compat_and_mismatch():
    df = make_synthetic(5, 5)
    c = FeatureContract.fit(df, "pre_balance")
    assert FeatureContract.from_dict(c.to_dict()).feature_set == "pre_balance"
    old = {"version": "1", "type_vocab": c.type_vocab, "feature_names": FeatureContract.fit(df, "base").feature_names}
    assert FeatureContract.from_dict(old).feature_set == "base"  # artifacts written before feature sets
    bad = {**c.to_dict(), "feature_names": c.feature_names[:-1]}
    assert_raises(ValueError, FeatureContract.from_dict, bad)
    assert "oldbalanceOrg" in str(assert_raises(KeyError, build_features, df.drop(columns=["oldbalanceOrg"]), c))


# ---------------- loading ----------------
def test_load_dataset_reads_only_the_columns_of_the_feature_set(tmp_path):
    f = tmp_path / "d.csv"
    make_synthetic(5, 5).to_csv(f, index=False)
    assert set(load_dataset(f).columns) == {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}
    assert set(load_dataset(f, feature_set="pre_balance").columns) == {"step", "type", "amount", "nameOrig", "nameDest", "isFraud", "oldbalanceOrg", "oldbalanceDest"}
    assert "isFlaggedFraud" not in load_dataset(f, feature_set="all_balance").columns
    assert_raises(DatasetError, load_dataset, f, None, "bogus")
    g = tmp_path / "g.csv"
    make_synthetic(5, 5).drop(columns=["oldbalanceDest"]).to_csv(g, index=False)
    assert_raises(SchemaError, load_dataset, g, None, "pre_balance")


# ---------------- training with a balance feature set ----------------
def test_training_records_feature_set_everywhere(tmp_path):
    s = run_training(make_synthetic(), kind="hist_gbm", params={"max_iter": 30}, out_root=tmp_path, run_id="p",
                     data_label="SYNTHETIC test fixture", use_mlflow=False, feature_set="pre_balance", make_plots=False)
    meta = load_artifact(Path(s["run_dir"]) / "primary").metadata
    assert s["feature_set"] == meta["feature_set"] == meta["feature_contract"]["feature_set"] == "pre_balance"
    assert meta["feature_contract"]["input_columns"][-2:] == ["oldbalanceOrg", "oldbalanceDest"]
    assert meta["provenance"]["feature_set"] == "pre_balance" and meta["model_version"] == "p"
    assert "pre_balance" in (Path(s["run_dir"]) / "report.md").read_text()


def test_all_balance_report_carries_the_diagnostic_warning(tmp_path):
    s = run_training(make_synthetic(), kind="logreg", out_root=tmp_path, run_id="a", data_label="SYNTHETIC test fixture",
                     use_mlflow=False, feature_set="all_balance", make_plots=False)
    txt = (Path(s["run_dir"]) / "report.md").read_text()
    assert "DIAGNOSTIC RUN" in txt and "post-transaction" in txt


# ---------------- API with a balance-feature model ----------------
def svc(tmp_path, fs):
    s = make_service(tmp_path, trained(fs))
    assert s.model.ready, s.model.reason
    return s


def test_model_info_exposes_input_fields(tmp_path):
    m = svc(tmp_path, "pre_balance").model_info()["model"]
    assert m["feature_set"] == "pre_balance" and m["input_fields"][-2:] == ["oldbalanceOrg", "oldbalanceDest"]
    assert m["model_version"] == "fs" and m["training"]["fit_seconds"] >= 0
    assert svc(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, "base").model_info()["model"]["input_fields"] == ["step", "type", "amount", "nameOrig", "nameDest"]


def test_balance_model_requires_balances_and_rejects_post_transaction_fields(tmp_path):
    s = svc(tmp_path, "pre_balance")
    e = assert_raises(AppError, s.predict_one, GOOD)  # base payload
    assert code_of(e) == "validation_error" and {d["field"] for d in e.details} == {"oldbalanceOrg", "oldbalanceDest"}
    assert_raises(AppError, s.predict_one, {**GOOD, **PRE, "newbalanceOrig": 0.0})  # post-transaction field refused
    assert_raises(AppError, s.predict_one, {**GOOD, **PRE, "oldbalanceOrg": -1.0})
    assert_raises(AppError, s.predict_one, {**GOOD, **PRE, "oldbalanceOrg": float("nan")})
    r = s.predict_one({**GOOD, **PRE})
    assert 0 <= r["score"] <= 1
    t = s.get_transaction(r["transaction_id"])
    assert t["oldbalance_orig"] == 5000.0 and t["newbalance_orig"] is None


def test_base_model_rejects_balance_fields_in_json(tmp_path):
    s = svc(tmp_path, "base")
    assert code_of(assert_raises(AppError, s.predict_one, {**GOOD, **PRE})) == "validation_error"
    assert s.predict_one(GOOD)["flagged"] in (True, False)


def test_batch_csv_columns_follow_the_model(tmp_path):
    full = make_synthetic(5, 5).head(12)
    s = svc(tmp_path, "pre_balance")
    out = s.predict_batch(csv_bytes(full))  # extra post-transaction columns present in the file are ignored
    assert out["rows"] == 12
    e = assert_raises(AppError, s.predict_batch, csv_bytes(full.drop(columns=["oldbalanceOrg"])))
    assert e.code == "missing_columns" and e.details["missing"] == ["oldbalanceOrg"]
    bad = full.copy()
    bad.loc[1, "oldbalanceDest"] = -5
    assert assert_raises(AppError, s.predict_batch, csv_bytes(bad)).code == "invalid_rows"
    assert s.repo.get_transaction(1)["oldbalance_dest"] is not None


def test_batch_scores_match_direct_model_scores_for_balance_model(tmp_path):
    s = svc(tmp_path, "pre_balance")
    full = make_synthetic(5, 5).head(15)
    out = s.predict_batch(csv_bytes(full))
    direct = s.model.loaded.predict_score(full)
    assert np.allclose([r["score"] for r in out["results"]], direct)


def test_balance_model_monitoring_skips_rows_without_balances(tmp_path):
    s = svc(tmp_path, "pre_balance")
    s.repo.insert_predictions([dict(step=1, type="TRANSFER", amount=1.0, nameOrig="Z", nameDest="C9", score=0.1,
                                    threshold=0.5, flagged=False, type_known=True)], "single", None,
                              {"kind": "x", "sha256": "y", "data_label": None})  # a row WITHOUT balances (older model)
    s.predict_batch(csv_bytes(make_synthetic(10, 20)))
    r = s.monitoring_status()
    assert r["status"] == "ok" and r["drift"]["rows_compared"] == 200  # the balance-less row was skipped


def test_database_created_before_balance_columns_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
      source TEXT NOT NULL, batch_id TEXT, step INTEGER NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL,
      name_orig TEXT NOT NULL, name_dest TEXT NOT NULL, score REAL NOT NULL, threshold REAL NOT NULL,
      flagged INTEGER NOT NULL, type_in_training_vocab INTEGER NOT NULL, model_kind TEXT NOT NULL,
      model_sha256 TEXT NOT NULL, model_data_label TEXT);
      INSERT INTO transactions VALUES (1,'2026-01-01T00:00:00+00:00','single',NULL,1,'T',5.0,'A','B',0.1,0.5,0,1,'k','h',NULL);""")
    con.commit(); con.close()
    repo = Repository(db)
    repo.init_schema()
    repo.init_schema()  # idempotent
    t = repo.get_transaction(1)
    assert t["oldbalance_orig"] is None and t["amount"] == 5.0
    (tid,) = repo.insert_predictions([dict(step=2, type="T", amount=1.0, nameOrig="C", nameDest="D", score=0.2, threshold=0.5,
                                           flagged=False, type_known=True, oldbalanceOrg=9.0)], "single", None,
                                     {"kind": "k", "sha256": "h", "data_label": None})
    assert repo.get_transaction(tid)["oldbalance_orig"] == 9.0
