"""Compare training runs side by side:  python -m ml.compare_runs ml/artifacts/<runA> ml/artifacts/<runB> ...
Shows each run's feature set and the primary model's validation vs test numbers. Never re-scores anything."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_rows(run_dirs) -> list[dict]:
    rows = []
    for d in run_dirs:
        f = Path(d) / "summary.json"
        if not f.is_file():
            raise FileNotFoundError(f"No summary.json in {Path(d).name}")
        s = json.loads(f.read_text(encoding="utf-8"))
        for role, m in s["models"].items():
            rows.append({"run": s["run_id"], "feature_set": s.get("feature_set", "base"), "role": role, "kind": m["kind"],
                         "data": s["data_label"], "val_pr_auc": m["validation"]["pr_auc"], "test_pr_auc": m["test"]["pr_auc"],
                         "test_precision": m["test"]["precision"], "test_recall": m["test"]["recall"],
                         "test_f1": m["test"]["f1"], "fp": m["test"]["fp"], "fn": m["test"]["fn"]})
    return rows


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ml.compare_runs <run_dir> [<run_dir> ...]", file=sys.stderr)
        return 2
    try:
        rows = load_rows(argv)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"{'run':34s} {'features':12s} {'role':9s} {'val PR-AUC':>10s} {'test PR-AUC':>11s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'FP':>6s} {'FN':>6s}")
    for r in rows:
        print(f"{r['run'][:34]:34s} {r['feature_set']:12s} {r['role']:9s} {_f(r['val_pr_auc']):>10s} {_f(r['test_pr_auc']):>11s} "
              f"{_f(r['test_precision']):>7s} {_f(r['test_recall']):>7s} {_f(r['test_f1']):>7s} {r['fp']:>6d} {r['fn']:>6d}")
    print("Choose between runs using the VALIDATION column; the test column is for final reporting only. "
          "all_balance runs are diagnostic (post-transaction information).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
