"""Tests use a tiny SYNTHETIC frame with PaySim's column layout. Numbers are not real results."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.eda import inspect_dataset as ins  # noqa: E402
from ml.eda import make_eda_report as rep  # noqa: E402


def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({
        "step": [1, 1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [10.0, 500.0, 200.0, 300.0],
        "nameOrig": ["C1", "C2", "C2", "C3"],
        "oldbalanceOrg": [100.0, 500.0, 200.0, 300.0],
        "newbalanceOrig": [90.0, 0.0, 0.0, 300.0],
        "nameDest": ["M1", "C9", "C8", "C9"],
        "oldbalanceDest": [0.0, 0.0, 5.0, 0.0],
        "newbalanceDest": [0.0, 0.0, 205.0, 0.0],
        "isFraud": [0, 1, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 0],
    })


def test_compute_stats_basic():
    s = ins.compute_stats(tiny_df())
    assert s["rows"] == 4
    assert s["fraud_count"] == 2
    assert s["fraud_rate"] == 0.5
    assert s["imbalance_ratio"] == 1.0
    assert s["type_counts"]["TRANSFER"] == 2
    assert s["fraud_by_type"]["TRANSFER"]["fraud_count"] == 1
    assert s["duplicated_rows"] == 0
    assert s["unique_nameDest"] == 3
    assert s["dest_merchant_rows"] == 1


def test_missing_and_duplicates_detected():
    df = tiny_df()
    df.loc[0, "amount"] = None
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    s = ins.compute_stats(df)
    assert s["missing_per_column"]["amount"] == 1
    assert s["duplicated_rows"] == 1


def test_missing_optional_columns_do_not_crash():
    s = ins.compute_stats(tiny_df()[["step", "amount"]])
    assert "fraud_rate" not in s and "type_counts" not in s


def test_find_csv_errors_when_empty(tmp_path):
    try:
        ins.find_csv(tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_report_renders_only_from_stats(tmp_path):
    f = tmp_path / "x.csv"
    tiny_df().to_csv(f, index=False)
    s = ins.compute_stats(pd.read_csv(f))
    s.update(file_name="x.csv", file_size_mb=0.0, sha256=ins.sha256_of(f))
    md = rep.render(json.loads(json.dumps(s)))
    assert "Fraud rate: 50.000000%" in md and "TODO" in md
