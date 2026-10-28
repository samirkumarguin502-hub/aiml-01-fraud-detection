"""Render docs/eda/EDA_REPORT.md from docs/eda/dataset_stats.json.

Only numbers computed by inspect_dataset.py are written. Refuses to run if the stats
file is missing, so a report can never be produced without real data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "docs" / "eda" / "dataset_stats.json"
REPORT = ROOT / "docs" / "eda" / "EDA_REPORT.md"


def render(s: dict) -> str:
    o = [f"# EDA Report (generated from `{s.get('file_name', '?')}`)", "",
         "All numbers below were computed by `ml/eda/inspect_dataset.py` on the file named above.",
         "Interpretation sections marked **TODO** are for the author to write after reviewing them.",
         "", "## Provenance", f"- File: `{s.get('file_name')}`  ({s.get('file_size_mb')} MB)",
         f"- SHA-256: `{s.get('sha256')}`", "", "## Shape",
         f"- Rows: {s['rows']:,}", f"- Columns ({len(s['columns'])}): {', '.join(s['columns'])}",
         "", "## Data quality",
         f"- Fully duplicated rows: {s['duplicated_rows']:,}",
         "- Missing values per column:"]
    o += [f"  - {c}: {n:,}" for c, n in s["missing_per_column"].items()]
    if "class_counts" in s:
        o += ["", "## Class balance", "| isFraud | rows |", "|---|---|"]
        o += [f"| {k} | {v:,} |" for k, v in s["class_counts"].items()]
        o.append(f"\n- Fraud rate: {s['fraud_rate']:.6%}")
        if s.get("imbalance_ratio") is not None:
            o.append(f"- Imbalance ratio (non-fraud : fraud): {s['imbalance_ratio']:.1f} : 1")
    if "fraud_by_type" in s:
        o += ["", "## Fraud by transaction type",
              "| type | transactions | fraud | fraud rate |", "|---|---|---|---|"]
        o += [f"| {t} | {r['transactions']:,} | {r['fraud_count']:,} | {r['fraud_rate']:.4%} |"
              for t, r in s["fraud_by_type"].items()]
    if "step" in s:
        o += ["", "## Time", f"- step min={s['step']['min']}, max={s['step']['max']}, "
              f"unique={s['step']['unique']}"]
    if "amount" in s:
        o += ["", "## Amount summary"] + [f"- {k}: {v:,.2f}" for k, v in s["amount"].items()]
        if "amount_fraud_total" in s:
            o.append(f"- Total amount in fraud rows: {s['amount_fraud_total']:,.2f} "
                     f"of {s['amount_total']:,.2f} overall")
    o += ["", "## Entities"]
    for col in ("nameOrig", "nameDest"):
        if f"unique_{col}" in s:
            o.append(f"- Unique {col}: {s[f'unique_{col}']:,}")
    if "dest_merchant_rows" in s:
        o.append(f"- Rows with merchant destination (M-prefix): {s['dest_merchant_rows']:,}")
    if "flagged_rows" in s:
        o += ["", "## Existing rule flag (isFlaggedFraud)", f"- Flagged rows: {s['flagged_rows']:,}"]
        if "flagged_and_fraud" in s:
            o.append(f"- Flagged and actually fraud: {s['flagged_and_fraud']:,}")
    if "balance_leakage_check" in s:
        o += ["", "## Balance-column leakage check (facts only)",
              "| group | rows | newbalanceOrig==0 | oldbalanceDest==0 | newbalanceDest==0 | orig balance unchanged |",
              "|---|---|---|---|---|---|"]
        for g, r in s["balance_leakage_check"].items():
            o.append(f"| {g} | {r['rows']:,} | {r['share_newbalanceOrig_zero']:.2%} | "
                     f"{r['share_oldbalanceDest_zero']:.2%} | {r['share_newbalanceDest_zero']:.2%} | "
                     f"{r['share_orig_balance_unchanged']:.2%} |")
    o += ["", "## Author's interpretation", "- **TODO:** leakage decision for balance columns",
          "- **TODO:** split strategy and justification", "- **TODO:** subsampling decision"]
    return "\n".join(o) + "\n"


def main() -> None:
    if not STATS.exists():
        sys.exit(f"{STATS} not found. Run `python ml/eda/inspect_dataset.py` on the real CSV first.")
    REPORT.write_text(render(json.loads(STATS.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
