"""Evaluation plots (matplotlib, headless). Every figure title carries the data label so a plot made from
synthetic data can never be mistaken for a real result. Returns the files written; [] if matplotlib is missing."""
from __future__ import annotations

from pathlib import Path


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_all(out_dir: Path, *, label: str, test_metrics: dict, curves: dict | None, baseline_curves: dict | None,
             importance: dict | None, model_kind: str) -> list[str]:
    plt = _plt()
    if plt is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    tag = f"{label} | held-out test set"

    # confusion matrix
    cm = [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], ["not fraud", "fraud"]); ax.set_yticks([0, 1], ["not fraud", "fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold {test_metrics['threshold']:.4f}\n{tag}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=130); plt.close(fig)
    written.append("confusion_matrix.png")

    if curves:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot(curves["pr"]["recall"], curves["pr"]["precision"], label=model_kind)
        if baseline_curves:
            ax.plot(baseline_curves["pr"]["recall"], baseline_curves["pr"]["precision"], "--", label="logreg baseline")
        ax.axhline(curves["prevalence"], color="grey", lw=0.8, ls=":", label="prevalence")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        ax.set_title(f"Precision-recall curve\n{tag}", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "precision_recall_curve.png", dpi=130); plt.close(fig)
        written.append("precision_recall_curve.png")
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], label=model_kind)
        if baseline_curves:
            ax.plot(baseline_curves["roc"]["fpr"], baseline_curves["roc"]["tpr"], "--", label="logreg baseline")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
        ax.set_title(f"ROC curve\n{tag}", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=130); plt.close(fig)
        written.append("roc_curve.png")

    if importance:
        names, vals = importance["features"][::-1], importance["values"][::-1]
        fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(names) + 1.4))
        ax.barh(names, vals, color="#3fb7a6")
        ax.set_xlabel(f"Importance ({importance['type']})")
        ax.set_title(f"Feature importance, top {len(names)}\n{label}", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=130); plt.close(fig)
        written.append("feature_importance.png")
    return written
