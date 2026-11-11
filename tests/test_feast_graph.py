"""Feast-support and graph tests. SYNTHETIC data. Feast itself, PyTorch and PyG are NOT exercised."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import assert_raises  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

from ml.feast_support import (EPOCH, FEAST_FEATURES, FeastSupportError, build_feature_table,  # noqa: E402
                              feast_status, point_in_time_lookup, step_to_timestamp, write_feature_table)
from ml.graph.build import (GraphInputError, dest_graph_features, evaluate_graph_baseline,  # noqa: E402
                            graph_status, graph_summary, validate_graph_input)
from ml.pipeline.features import FeatureContract, build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def repeated_senders() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for step in range(1, 41):
        for i in range(12):
            rows.append(dict(step=step, type="TRANSFER", amount=float(rng.integers(10, 1000)),
                             nameOrig=f"C{rng.integers(1, 15)}", nameDest=f"C{100 + rng.integers(1, 10)}",
                             isFraud=int(rng.random() < 0.1)))
    return pd.DataFrame(rows)


# ---- Feast support ----
def test_feast_features_match_training_contract_exactly():
    df = repeated_senders()
    contract = FeatureContract.fit(df)
    expected = build_features(df, contract)[["orig_prev_txn_count", "orig_prev_amount_mean"]]
    got = point_in_time_lookup(df, build_feature_table(df))
    pd.testing.assert_frame_equal(expected, got, check_exact=False, rtol=1e-12)
    assert expected["orig_prev_txn_count"].max() > 3  # the fixture really exercises repeated senders


def test_feast_table_never_exposes_same_or_future_steps():
    df = pd.DataFrame(dict(step=[1, 1, 2], type="T", amount=[10.0, 20.0, 30.0], nameOrig="C1", nameDest="C2"))
    got = point_in_time_lookup(df, build_feature_table(df))
    assert got["orig_prev_txn_count"].tolist() == [0.0, 0.0, 2.0]
    assert got["orig_prev_amount_mean"].tolist() == [0.0, 0.0, 15.0]


def test_feast_timestamps_are_a_documented_mapping_not_dates():
    t = build_feature_table(repeated_senders())
    assert t["event_timestamp"].min() >= EPOCH and step_to_timestamp(0) == EPOCH
    assert set(FEAST_FEATURES) <= set(t.columns)


def test_feast_definitions_match_contract_and_config_is_local():
    defs = (ROOT / "feature_repo" / "definitions.py").read_text()
    fields = set(re.findall(r'Field\(name="([a-z_]+)"', defs))
    assert fields == set(FEAST_FEATURES) and 'join_keys=["nameOrig"]' in defs
    assert "orig_prev_amount_mean" not in defs  # derived at serving, not stored
    cfg = (ROOT / "feature_repo" / "feature_store.yaml").read_text()
    assert "provider: local" in cfg and "type: sqlite" in cfg and "type: file" in cfg
    assert not re.search(r"(password|secret|token|key_id)", cfg, re.I)
    contract_names = set(FeatureContract.fit(make_synthetic(3, 3)).feature_names)
    assert {"orig_prev_txn_count", "orig_prev_amount_mean"} <= contract_names


def test_feast_optional_when_missing():
    st = feast_status()
    assert set(st) == {"library_installed", "note"}
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        e = assert_raises(FeastSupportError, write_feature_table, repeated_senders(), Path("x/y.parquet"))
        assert "pyarrow" in str(e)
    assert_raises(FeastSupportError, build_feature_table, pd.DataFrame({"a": [1]}))


# ---- graph ----
def test_graph_status_reports_optional_gnn():
    s = graph_status()
    assert "optional CPU PyG" in s["gnn_training"] and s["core_demo_requires_graph"] is False
    assert isinstance(s["torch_geometric_installed"], bool)


def test_graph_input_validation_and_identifier_honesty():
    assert_raises(GraphInputError, validate_graph_input, pd.DataFrame({"step": [1]}))
    assert_raises(GraphInputError, validate_graph_input, pd.DataFrame({"step": [], "nameOrig": [], "nameDest": []}))
    f = validate_graph_input(repeated_senders().assign(deviceId="d1"))
    assert f["identifier_columns_used"] == ["nameOrig", "nameDest"]
    assert f["unsupported_identifier_columns_present_but_unused"] == ["deviceId"]
    assert "none are constructed" in f["device_card_relationships"]


def test_graph_summary_on_known_structure():
    df = pd.DataFrame(dict(step=[1, 2, 3, 4], nameOrig=["A", "A", "B", "X"], nameDest=["B", "B", "C", "Y"]))
    s = graph_summary(df)
    assert s["nodes"] == 5 and s["distinct_edges"] == 3 and s["transactions"] == 4
    assert s["weak_components"] == 2 and s["largest_component_nodes"] == 3  # {A,B,C} and {X,Y}
    assert s["max_in_degree"] == 1 and s["max_out_degree"] == 1 and abs(s["sender_repeat_fraction"] - 1 / 3) < 1e-12


def test_dest_features_are_causal_and_correct():
    df = pd.DataFrame(dict(step=[1, 1, 2, 3, 3], nameOrig=["A", "B", "A", "C", "A"], nameDest=["D"] * 5))
    f = dest_graph_features(df)
    assert f["dest_prev_txn_count"].tolist() == [0, 0, 2, 3, 3]
    assert f["dest_prev_unique_senders"].tolist() == [0, 0, 2, 2, 2]  # C first appears at step 3 -> not counted yet
    df2 = df.copy()
    df2.loc[df2["step"] >= 3, "nameDest"] = "Z"  # change the future
    f2 = dest_graph_features(df2)
    assert f2.loc[df["step"] < 3, :].equals(f.loc[df["step"] < 3, :])


def test_graph_baseline_evaluation_uses_test_range_and_reports_insufficiency():
    df = repeated_senders()
    r = evaluate_graph_baseline(df)
    assert r["status"] == "ok" and set(r["features"]) == {"dest_prev_txn_count", "dest_prev_unique_senders"}
    assert 0 <= r["features"]["dest_prev_txn_count"]["pr_auc"] <= 1 and "not validated" in r["combination_with_xgboost"]
    late = df.assign(isFraud=np.where(df["step"] > 25, 0, df["isFraud"]))  # test range has no fraud
    assert evaluate_graph_baseline(late)["status"] == "insufficient_data"
    assert_raises(GraphInputError, evaluate_graph_baseline, df.drop(columns=["isFraud"]))
