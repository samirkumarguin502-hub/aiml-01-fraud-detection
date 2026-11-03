"""SYNTHETIC fixtures only (PaySim-like layout). Nothing here is a real result."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.pipeline.features import (FORBIDDEN_FEATURE_COLUMNS, UNKNOWN_TYPE,  # noqa: E402
                                  FeatureContract, build_features)
from ml.pipeline.schema import SchemaError, validate_schema  # noqa: E402
from ml.pipeline.split import time_split  # noqa: E402


def synth(n_steps=20) -> pd.DataFrame:
    rows = []
    for s in range(1, n_steps + 1):
        rows.append(dict(step=s, type="TRANSFER", amount=100.0 + s, nameOrig="C1", nameDest="C9",
                         oldbalanceOrg=5.0, newbalanceOrig=0.0, oldbalanceDest=0.0,
                         newbalanceDest=0.0, isFraud=int(s % 7 == 0), isFlaggedFraud=0))
        rows.append(dict(step=s, type="PAYMENT", amount=10.0, nameOrig=f"C{s+100}", nameDest="M1",
                         oldbalanceOrg=50.0, newbalanceOrig=40.0, oldbalanceDest=0.0,
                         newbalanceDest=0.0, isFraud=0, isFlaggedFraud=0))
    return pd.DataFrame(rows)


def test_schema_ok_reports_observed_types():
    assert validate_schema(synth())["types_observed"] == ["PAYMENT", "TRANSFER"]


def test_schema_missing_column_and_bad_target():
    for bad in (synth().drop(columns=["amount"]),
                synth().assign(isFraud=2),
                synth().assign(isFraud=0),
                synth().assign(amount=-1.0)):
        try:
            validate_schema(bad)
        except SchemaError:
            continue
        raise AssertionError("expected SchemaError")


def test_type_spellings_are_not_merged():
    df = synth()
    df.loc[0, "type"] = "CASH-OUT"
    df.loc[1, "type"] = "CASH_OUT"
    obs = validate_schema(df)["types_observed"]
    assert "CASH-OUT" in obs and "CASH_OUT" in obs


def test_no_leakage_columns_in_features():
    df = synth()
    c = FeatureContract.fit(df)
    X = build_features(df, c)
    assert not set(X.columns) & set(FORBIDDEN_FEATURE_COLUMNS)
    assert list(X.columns) == c.feature_names


def test_features_ignore_leakage_values():
    df = synth()
    c = FeatureContract.fit(df)
    a = build_features(df, c)
    b = build_features(df.assign(newbalanceOrig=999.0, oldbalanceDest=7.0, isFlaggedFraud=1,
                                 isFraud=1 - df["isFraud"]), c)
    pd.testing.assert_frame_equal(a, b)


def test_history_is_causal_future_changes_do_not_leak_back():
    df = synth()
    c = FeatureContract.fit(df)
    base = build_features(df, c)
    changed = df.copy()
    changed.loc[changed["step"] >= 15, "amount"] = 1e9  # alter the future
    after = build_features(changed, c)
    early = df["step"] < 15
    cols = ["orig_prev_txn_count", "orig_prev_amount_mean"]
    pd.testing.assert_frame_equal(base.loc[early, cols], after.loc[early, cols])


def test_history_excludes_same_step():
    df = pd.DataFrame(dict(step=[1, 1, 2], type="TRANSFER", amount=[10.0, 20.0, 30.0],
                           nameOrig="C1", nameDest="C2", isFraud=[0, 1, 0]))
    X = build_features(df, FeatureContract.fit(df))
    assert X["orig_prev_txn_count"].tolist() == [0.0, 0.0, 2.0]
    assert X["orig_prev_amount_mean"].tolist() == [0.0, 0.0, 15.0]


def test_unseen_type_goes_to_unknown_bucket():
    df = synth()
    c = FeatureContract.fit(df)
    new = df.head(2).assign(type="BRAND_NEW")
    X = build_features(new, c)
    assert X[f"type__{UNKNOWN_TYPE}"].tolist() == [1, 1]
    assert X["type__TRANSFER"].sum() == 0


def test_contract_roundtrip_and_version_check():
    c = FeatureContract.fit(synth())
    assert FeatureContract.from_dict(c.to_dict()).feature_names == c.feature_names
    try:
        FeatureContract.from_dict({**c.to_dict(), "version": "0"})
    except ValueError:
        return
    raise AssertionError("expected version mismatch error")


def test_time_split_is_strictly_ordered_and_disjoint():
    df = synth(30)
    tr, va, te, info = time_split(df)
    assert tr["step"].max() < va["step"].min() <= va["step"].max() < te["step"].min()
    assert len(tr) + len(va) + len(te) == len(df)
    assert info["train_max_step"] == tr["step"].max()


def test_time_split_rejects_too_few_steps():
    try:
        time_split(synth(2))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
