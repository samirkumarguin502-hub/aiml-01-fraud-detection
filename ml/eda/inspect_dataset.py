"""Inspect the raw PaySim CSV and record facts only (no assumptions).

Run from the project root:
    python ml/eda/inspect_dataset.py

Writes:
    docs/eda/dataset_inspection.txt   human-readable output
    docs/eda/dataset_stats.json       machine-readable stats (input to make_eda_report.py)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
OUT_DIR = ROOT / "docs" / "eda"

BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def find_csv(raw_dir: Path = RAW_DIR) -> Path:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Download the dataset first.")
    return files[0]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute facts from the dataframe. Missing optional columns are skipped, not guessed."""
    s: dict = {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 1),
        "head": json.loads(df.head().to_json(orient="records")),
    }
    if "isFraud" in df.columns:
        counts = df["isFraud"].value_counts()
        s["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        s["fraud_count"] = int(counts.get(1, 0))
        s["fraud_rate"] = float(df["isFraud"].mean())
        s["imbalance_ratio"] = (
            float(counts[0] / counts[1]) if 0 in counts.index and 1 in counts.index else None
        )
    if "type" in df.columns:
        s["type_counts"] = {k: int(v) for k, v in df["type"].value_counts().items()}
        if "isFraud" in df.columns:
            g = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
            s["fraud_by_type"] = {
                t: {"transactions": int(r["count"]), "fraud_count": int(r["sum"]),
                    "fraud_rate": float(r["mean"])}
                for t, r in g.iterrows()
            }
    if "step" in df.columns:
        s["step"] = {"min": int(df["step"].min()), "max": int(df["step"].max()),
                     "unique": int(df["step"].nunique())}
    if "amount" in df.columns:
        d = df["amount"].describe()
        s["amount"] = {k: float(v) for k, v in d.items()}
        if "isFraud" in df.columns:
            s["amount_fraud_total"] = float(df.loc[df["isFraud"] == 1, "amount"].sum())
            s["amount_total"] = float(df["amount"].sum())
    for col in ("nameOrig", "nameDest"):
        if col in df.columns:
            s[f"unique_{col}"] = int(df[col].nunique())
    if "nameDest" in df.columns:
        s["dest_merchant_rows"] = int(df["nameDest"].astype(str).str.startswith("M").sum())
    if "isFlaggedFraud" in df.columns:
        s["flagged_rows"] = int(df["isFlaggedFraud"].sum())
        if "isFraud" in df.columns:
            s["flagged_and_fraud"] = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())
    # Leakage evidence: do balance columns behave differently for fraud rows?
    if "isFraud" in df.columns and all(c in df.columns for c in BALANCE_COLS):
        leak = {}
        for label, mask in (("fraud", df["isFraud"] == 1), ("non_fraud", df["isFraud"] == 0)):
            sub = df[mask]
            n = len(sub)
            if n == 0:
                continue
            leak[label] = {
                "rows": int(n),
                "share_newbalanceOrig_zero": float((sub["newbalanceOrig"] == 0).mean()),
                "share_oldbalanceDest_zero": float((sub["oldbalanceDest"] == 0).mean()),
                "share_newbalanceDest_zero": float((sub["newbalanceDest"] == 0).mean()),
                "share_orig_balance_unchanged": float(
                    (sub["oldbalanceOrg"] == sub["newbalanceOrig"]).mean()),
            }
        s["balance_leakage_check"] = leak
    return s


def format_text(s: dict, file_name: str, size_mb: float, sha: str) -> str:
    L = [f"File: {file_name}", f"Size on disk: {size_mb:.1f} MB", f"SHA-256: {sha}",
         f"\n== SHAPE ==\nrows={s['rows']:,}  columns={len(s['columns'])}",
         "\n== COLUMNS AND DTYPES =="]
    L += [f"{c}: {s['dtypes'][c]}" for c in s["columns"]]
    L.append("\n== FIRST 5 ROWS ==")
    L += [json.dumps(r) for r in s["head"]]
    L.append("\n== MISSING VALUES (per column) ==")
    L += [f"{c}: {n}" for c, n in s["missing_per_column"].items()]
    L.append(f"\n== DUPLICATES ==\nfully duplicated rows: {s['duplicated_rows']:,}")
    if "class_counts" in s:
        L.append("\n== CLASS BALANCE (isFraud) ==")
        L += [f"{k}: {v:,}" for k, v in s["class_counts"].items()]
        L.append(f"fraud rate: {s['fraud_rate']:.6%}")
        if s["imbalance_ratio"] is not None:
            L.append(f"imbalance ratio (non-fraud : fraud) = {s['imbalance_ratio']:.1f} : 1")
    else:
        L.append("\nNo 'isFraud' column found - check the column list above.")
    if "type_counts" in s:
        L.append("\n== TRANSACTION TYPES ==")
        L += [f"{k}: {v:,}" for k, v in s["type_counts"].items()]
    if "fraud_by_type" in s:
        L.append("\n== FRAUD BY TYPE ==")
        for t, r in s["fraud_by_type"].items():
            L.append(f"{t}: transactions={r['transactions']:,} fraud={r['fraud_count']:,} "
                     f"rate={r['fraud_rate']:.6%}")
    if "step" in s:
        L.append(f"\n== TIME STEP ==\nmin={s['step']['min']} max={s['step']['max']} "
                 f"unique={s['step']['unique']}")
    if "amount" in s:
        L.append("\n== AMOUNT SUMMARY ==")
        L += [f"{k}: {v:,.2f}" for k, v in s["amount"].items()]
    for col in ("nameOrig", "nameDest"):
        if f"unique_{col}" in s:
            L.append(f"\nunique {col}: {s[f'unique_{col}']:,}")
    if "dest_merchant_rows" in s:
        L.append(f"rows with merchant destination (nameDest starts with M): "
                 f"{s['dest_merchant_rows']:,}")
    if "flagged_rows" in s:
        L.append(f"\nisFlaggedFraud == 1 rows: {s['flagged_rows']:,}")
        if "flagged_and_fraud" in s:
            L.append(f"isFlaggedFraud == 1 AND isFraud == 1: {s['flagged_and_fraud']:,}")
    if "balance_leakage_check" in s:
        L.append("\n== BALANCE-COLUMN LEAKAGE CHECK (facts only) ==")
        for label, r in s["balance_leakage_check"].items():
            L.append(f"{label} (rows={r['rows']:,}): " + ", ".join(
                f"{k}={v:.4%}" for k, v in r.items() if k != "rows"))
    L.append(f"\nMemory in pandas: {s['memory_mb']:.0f} MB")
    return "\n".join(L)


def main() -> None:
    try:
        path = find_csv()
    except FileNotFoundError as e:
        sys.exit(str(e))
    df = pd.read_csv(path)
    stats = compute_stats(df)
    stats["file_name"] = path.name
    stats["file_size_mb"] = round(path.stat().st_size / 1e6, 1)
    stats["sha256"] = sha256_of(path)
    text = format_text(stats, path.name, stats["file_size_mb"], stats["sha256"])
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_inspection.txt").write_text(text, encoding="utf-8")
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
