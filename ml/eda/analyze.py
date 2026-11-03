"""Extended EDA on the REAL file: balance consistency, amount stats by class, fraud per step, a single-rule
diagnostic, plots, and docs/eda_report.md.

    python -m ml.eda.analyze                       # first *.csv in ml/data/raw/
    python -m ml.eda.analyze --data <csv> [--nrows N] [--out-dir docs/eda]

Everything is computed from the file you point it at; nothing is assumed. Reads all 11 columns (EDA only; training
never loads the balance columns unless a balance feature set is selected).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.eda.inspect_dataset import RAW_DIR, compute_stats, find_csv, sha256_of

ROOT = Path(__file__).resolve().parents[2]
BAL = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
TOL = 0.01


def balance_analysis(df: pd.DataFrame) -> dict:
    """Balance behaviour per class and a single-rule diagnostic. Requires the four balance columns + isFraud."""
    need = ("type", "amount", "isFraud", "nameDest") + BAL
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for balance analysis: {miss}")
    d = df
    err_o = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
    err_d = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
    emptied = (d["oldbalanceOrg"] > 0) & (d["amount"] >= d["oldbalanceOrg"] - 0.005)  # PRE-transaction information
    merchant = d["nameDest"].astype(str).str.startswith("M")
    out: dict = {"definitions": {
        "orig_balance_error": "newbalanceOrig + amount - oldbalanceOrg (0 when the origin account is debited exactly)",
        "dest_balance_error": "oldbalanceDest + amount - newbalanceDest (0 when the destination is credited exactly)",
        "orig_emptied": "oldbalanceOrg > 0 and amount >= oldbalanceOrg (known BEFORE the transaction)",
        "tolerance": TOL}, "by_class": {}, "by_type": {}}

    def block(mask: pd.Series) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"rows": 0}
        s = d[mask]
        return {"rows": n,
                "share_oldbalanceOrg_zero": float((s["oldbalanceOrg"] == 0).mean()),
                "share_oldbalanceDest_zero": float((s["oldbalanceDest"] == 0).mean()),
                "share_newbalanceOrig_zero": float((s["newbalanceOrig"] == 0).mean()),
                "share_newbalanceDest_zero": float((s["newbalanceDest"] == 0).mean()),
                "share_orig_emptied": float(emptied[mask].mean()),
                "share_orig_error_nonzero": float((err_o[mask].abs() > TOL).mean()),
                "share_dest_error_nonzero": float((err_d[mask].abs() > TOL).mean()),
                "share_dest_is_merchant": float(merchant[mask].mean())}
    for label, m in (("fraud", d["isFraud"] == 1), ("non_fraud", d["isFraud"] == 0)):
        out["by_class"][label] = block(m)
    for t in sorted(d["type"].astype(str).unique()):
        out["by_type"][t] = {"fraud": block((d["type"].astype(str) == t) & (d["isFraud"] == 1)),
                             "non_fraud": block((d["type"].astype(str) == t) & (d["isFraud"] == 0))}

    # Single-rule diagnostic: "orig_emptied AND type is one that contains fraud" as a classifier.
    fraud_types = sorted(d.loc[d["isFraud"] == 1, "type"].astype(str).unique())
    rule = emptied & d["type"].astype(str).isin(fraud_types)
    y = d["isFraud"] == 1
    tp, fp, fn = int((rule & y).sum()), int((rule & ~y).sum()), int((~rule & y).sum())
    out["single_rule_diagnostic"] = {
        "rule": "orig_emptied AND type in {types that contain any fraud in THIS file}", "types_used": fraud_types,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) else None, "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "reading": ("If precision and recall are both high, one pre-transaction rule explains most fraud: the data "
                    "generator's pattern, not realistic difficulty. Treat any model that uses balances accordingly.")}
    return out


def amount_by_class(df: pd.DataFrame) -> dict:
    out = {}
    for label, v in (("fraud", 1), ("non_fraud", 0)):
        a = df.loc[df["isFraud"] == v, "amount"]
        out[label] = {k: float(x) for k, x in a.describe().items()} if len(a) else {"count": 0.0}
    return out


def fraud_per_step(df: pd.DataFrame) -> dict:
    g = df.groupby("step")["isFraud"].agg(rows="count", fraud="sum")
    return {"steps": int(len(g)), "rows_per_step": {"min": int(g["rows"].min()), "median": float(g["rows"].median()),
                                                     "max": int(g["rows"].max())},
            "fraud_per_step": {"min": int(g["fraud"].min()), "median": float(g["fraud"].median()),
                               "max": int(g["fraud"].max())},
            "steps_with_zero_fraud": int((g["fraud"] == 0).sum()),
            "series": {"step": [int(i) for i in g.index], "rows": [int(v) for v in g["rows"]],
                       "fraud": [int(v) for v in g["fraud"]]}}


def make_plots(df: pd.DataFrame, out_dir: Path, label: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    g = df.groupby("type")["isFraud"].agg(["count", "sum"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(g.index.astype(str), g["sum"] / g["count"] * 100, color="#d9534f")
    ax.set_ylabel("Fraud rate (%)"); ax.set_xlabel("Transaction type"); ax.set_title(f"Fraud rate by type\n{label}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "fraud_rate_by_type.png", dpi=120); plt.close(fig); written.append("fraud_rate_by_type.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for v, name, c in ((0, "not fraud", "#3fb7a6"), (1, "fraud", "#d9534f")):
        a = df.loc[df["isFraud"] == v, "amount"]
        if len(a):
            ax.hist(np.log10(a.clip(lower=0) + 1), bins=50, alpha=0.6, density=True, label=name, color=c)
    ax.set_xlabel("log10(amount + 1)"); ax.set_ylabel("Density"); ax.legend(); ax.set_title(f"Amount distribution by class\n{label}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "amount_by_class.png", dpi=120); plt.close(fig); written.append("amount_by_class.png")
    s = df.groupby("step")["isFraud"].agg(["count", "sum"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(s.index, s["sum"], color="#d9534f")
    ax.set_xlabel("step (1 step = 1 hour)"); ax.set_ylabel("Fraud transactions"); ax.set_title(f"Fraud count per step\n{label}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "fraud_per_step.png", dpi=120); plt.close(fig); written.append("fraud_per_step.png")
    c = df["isFraud"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.bar(["not fraud", "fraud"], [int(c.get(0, 0)), int(c.get(1, 0))], color=["#3fb7a6", "#d9534f"]); ax.set_yscale("log")
    ax.set_ylabel("Rows (log scale)"); ax.set_title(f"Class balance\n{label}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "class_balance.png", dpi=120); plt.close(fig); written.append("class_balance.png")
    return written


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.2%}"


def render_markdown(info: dict, stats: dict, bal: dict, amt: dict, steps: dict, plots: list[str]) -> str:
    o = [f"# EDA report: {info['file_name']}", "",
         f"Generated by `python -m ml.eda.analyze` from the file below. Every number is computed from that file.", "",
         f"- File: `{info['file_name']}` ({info['file_size_mb']} MB), SHA-256 `{info['sha256']}`",
         f"- Rows: {stats['rows']:,}; columns: {len(stats['columns'])} ({', '.join(stats['columns'])})",
         f"- Duplicated rows: {stats['duplicated_rows']:,}; missing values: {sum(stats['missing_per_column'].values()):,}",
         f"- Fraud rows: {stats.get('fraud_count', 0):,} ({stats.get('fraud_rate', 0):.4%}); imbalance ratio "
         f"{stats.get('imbalance_ratio') or float('nan'):,.1f} : 1", "",
         "PaySim is a **simulation** of mobile-money transactions, not real bank customer data.", "",
         "## Transaction types (observed spellings)", "", "| type | transactions | fraud | fraud rate |", "|---|---|---|---|"]
    o += [f"| {t} | {v['transactions']:,} | {v['fraud_count']:,} | {v['fraud_rate']:.4%} |" for t, v in stats.get("fraud_by_type", {}).items()]
    o += ["", "## Amount by class", "", "| class | count | mean | median | max |", "|---|---|---|---|---|"]
    o += [f"| {k} | {int(v.get('count', 0)):,} | {v.get('mean', float('nan')):,.2f} | {v.get('50%', float('nan')):,.2f} | {v.get('max', float('nan')):,.2f} |" for k, v in amt.items()]
    o += ["", "## Time steps", "",
          f"{steps['steps']} distinct steps; rows per step min/median/max {steps['rows_per_step']['min']:,}/{steps['rows_per_step']['median']:,.0f}/{steps['rows_per_step']['max']:,}; "
          f"fraud per step min/median/max {steps['fraud_per_step']['min']}/{steps['fraud_per_step']['median']:.0f}/{steps['fraud_per_step']['max']}; "
          f"steps with zero fraud: {steps['steps_with_zero_fraud']}.", "",
          "## Balance behaviour by class", "", "| measure | fraud | not fraud |", "|---|---|---|"]
    f, n = bal["by_class"]["fraud"], bal["by_class"]["non_fraud"]
    for k, lab in (("share_oldbalanceOrg_zero", "origin balance before = 0"), ("share_orig_emptied", "amount empties origin account (known before the transaction)"),
                   ("share_newbalanceOrig_zero", "origin balance after = 0 (post-transaction)"), ("share_oldbalanceDest_zero", "destination balance before = 0"),
                   ("share_newbalanceDest_zero", "destination balance after = 0 (post-transaction)"),
                   ("share_orig_error_nonzero", "origin balance error != 0 (post-transaction)"), ("share_dest_error_nonzero", "destination balance error != 0 (post-transaction)"),
                   ("share_dest_is_merchant", "receiver is a merchant (M…)")):
        o.append(f"| {lab} | {_pct(f.get(k))} | {_pct(n.get(k))} |")
    d = bal["single_rule_diagnostic"]
    o += ["", "## Leakage / generator-rule diagnostic", "",
          f"Rule: `{d['rule']}` (types: {', '.join(d['types_used']) or 'none'}).", "",
          f"On this file: TP {d['tp']:,}, FP {d['fp']:,}, FN {d['fn']:,}; precision {_pct(d['precision'])}, recall {_pct(d['recall'])}.", "",
          d["reading"], "",
          "Post-transaction balances (`newbalance*` and the error terms) are not known when a transaction is scored and, in PaySim, "
          "differ systematically for fraud. The `all_balance` feature set exists only to measure how optimistic those make a model.", "",
          "## Plots", ""] + ([f"![{p}](eda/plots/{p})" for p in plots] or ["(matplotlib not installed: no plots)"])
    o += ["", "## What this report does not establish", "",
          "Licence/usage terms and download provenance must be recorded by you in `docs/paysim_provenance.md`. Nothing here says how "
          "a model will perform; see `docs/real_training_report.md`."]
    return "\n".join(o) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extended EDA on the real PaySim file.")
    ap.add_argument("--data", default=None, help="CSV path (default: first *.csv in ml/data/raw)")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "eda"))
    ap.add_argument("--report", default=str(ROOT / "docs" / "eda_report.md"))
    a = ap.parse_args(argv)
    try:
        path = Path(a.data) if a.data else find_csv(RAW_DIR)
        df = pd.read_csv(path, nrows=a.nrows)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, SystemExit) as e:
        print(f"ERROR: could not read the dataset ({type(e).__name__}). Put the PaySim CSV in ml/data/raw/ (docs/paysim_provenance.md).", file=sys.stderr)
        return 2
    try:
        stats = compute_stats(df)
        bal, amt, steps = balance_analysis(df), amount_by_class(df), fraud_per_step(df)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    out_dir = Path(a.out_dir)
    plots = make_plots(df, out_dir / "plots", path.name)
    info = {"file_name": path.name, "file_size_mb": round(path.stat().st_size / 1e6, 1), "sha256": sha256_of(path)}
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "balance_analysis.json").write_text(json.dumps({"info": info, "balance": bal, "amount": amt, "steps": {k: v for k, v in steps.items() if k != "series"}}, indent=2), encoding="utf-8")
    Path(a.report).write_text(render_markdown(info, stats, bal, amt, steps, plots), encoding="utf-8")
    print(f"wrote {Path(a.report).name}, balance_analysis.json and {len(plots)} plot(s) for {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
