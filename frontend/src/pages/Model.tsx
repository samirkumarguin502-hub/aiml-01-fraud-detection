import { useApi } from "../api-context.tsx";
import { CurveChart, HBarChart, LineChart } from "../components/charts.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { Async, Badge, Banner, Card, EmptyState, ScoreNote, SyntheticBanner } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtDate, fmtInt, fmtMetric, fmtRate, fmtScore, shortHash } from "../lib/format.ts";
import type { SetMetrics } from "../lib/types.ts";

function MetricsTable({ rows }: { rows: [string, SetMetrics | undefined][] }) {
  return (
    <div className="tablewrap"><table><caption className="small muted" style={{ textAlign: "left" }}>Metrics at the active threshold. PR-AUC and ROC-AUC are threshold-free.</caption>
      <thead><tr><th>Set</th><th className="num">PR-AUC</th><th className="num">ROC-AUC</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Fraud rows</th><th className="num">TP</th><th className="num">FP</th><th className="num">FN</th><th className="num">TN</th></tr></thead>
      <tbody>{rows.map(([n, m]) => m ? <tr key={n}><td>{n}</td><td className="num">{fmtMetric(m.pr_auc)}</td><td className="num">{fmtMetric(m.roc_auc)}</td><td className="num">{fmtMetric(m.precision)}</td><td className="num">{fmtMetric(m.recall)}</td><td className="num">{fmtMetric(m.f1)}</td><td className="num">{fmtInt(m.n_fraud)} / {fmtInt(m.n)}</td><td className="num">{fmtInt(m.tp)}</td><td className="num">{fmtInt(m.fp)}</td><td className="num">{fmtInt(m.fn)}</td><td className="num">{fmtInt(m.tn)}</td></tr> : null)}</tbody></table></div>
  );
}

export function Model() {
  const api = useApi();
  const [s, reload] = useAsync((sig) => api.model(sig), []);
  return (
    <>
      <PageHead title="Model performance & thresholds" sub="Read from the loaded model artifact. These are the training run's own held-out results, not live accuracy." actions={<button className="btn" onClick={reload}>Refresh</button>} />
      <Async state={s} reload={reload}>{(info) => {
        if (info.state !== "ready" || !info.model) return (
          <Card><EmptyState title={info.state === "invalid" ? "A model artifact was found but rejected" : "No trained model is loaded"}>{info.reason ?? ""} No performance metrics are shown because none exist for the running service.<br />Train one with <span className="mono">python -m ml.train --data &lt;csv&gt;</span> and set <span className="mono">MODEL_ARTIFACT_DIR</span>.</EmptyState></Card>
        );
        const m = info.model, mm = m.metrics, table = mm.test_threshold_table ?? [];
        return (
          <div className="grid">
            {mm.metrics_are_synthetic ? <SyntheticBanner origin={mm.origin} /> : null}
            <div className="grid g2">
              <Card title="Active model"><dl className="kv">
                <dt>Kind</dt><dd>{m.kind}</dd><dt>Fingerprint</dt><dd className="mono">{shortHash(m.sha256)}</dd><dt>Trained</dt><dd>{fmtDate(m.created_utc)}</dd>
                <dt>Metrics origin</dt><dd><Badge tone={mm.metrics_are_synthetic ? "warn" : "neutral"}>{mm.origin}</Badge></dd>
                <dt>Dataset SHA-256</dt><dd className="mono">{mm.dataset_sha256 ? shortHash(mm.dataset_sha256) + "…" : "none recorded"}</dd>
                <dt>Features</dt><dd>{m.feature_count} (contract v{m.feature_version})</dd><dt>Known types</dt><dd>{m.type_vocabulary.join(", ") || "n/a"}</dd></dl></Card>
              <Card title="Decision threshold"><p className="v" style={{ fontSize: "1.8rem", margin: 0 }}>{fmtScore(m.threshold.threshold)}</p>
                <p className="small muted">Chosen on validation data: {m.threshold.policy}.{m.threshold.fallback_to_max_f1 ? " The precision target could not be met, so max-F1 was used." : ""}</p>
                <p className="small muted">A higher threshold raises precision (fewer false alerts) and lowers recall (more missed fraud).</p><ScoreNote /></Card>
            </div>
            <Card title="Feature set and training">
              <dl className="kv"><dt>Feature set</dt><dd><Badge tone={m.feature_set === "all_balance" ? "bad" : "info"}>{m.feature_set ?? "base (older artifact)"}</Badge></dd>
                <dt>Inputs the API needs</dt><dd className="mono">{(m.input_fields ?? ["step", "type", "amount", "nameOrig", "nameDest"]).join(", ")}</dd>
                <dt>Model version</dt><dd className="mono">{m.model_version ?? "n/a"}</dd>
                <dt>Training time</dt><dd>{m.training?.fit_seconds !== undefined ? `${m.training.fit_seconds} s on ${m.training.cpu_count ?? "?"} CPU(s)` : "not recorded"}</dd>
                <dt>Peak memory</dt><dd>{m.training?.peak_memory_mb ? `${m.training.peak_memory_mb} MB` : "not recorded"}</dd></dl>
              {m.feature_set === "all_balance" ? <Banner tone="bad" title="Diagnostic model">It uses post-transaction balances that are not known when a transaction is scored. Its metrics are optimistic by construction.</Banner> : null}
            </Card>
            <Card title="Validation and held-out test"><MetricsTable rows={[["validation", mm.validation], ["test", mm.test]]} /></Card>
            {mm.test_classification_report ? (
              <Card title="Classification report (held-out test, at the active threshold)">
                <div className="tablewrap"><table><thead><tr><th>Class</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Support</th></tr></thead>
                  <tbody>{(["not_fraud", "fraud", "macro avg", "weighted avg"] as const).map((k) => { const r = mm.test_classification_report![k]; return <tr key={k}><td>{k.replace("_", " ")}</td><td className="num">{fmtMetric(r.precision)}</td><td className="num">{fmtMetric(r.recall)}</td><td className="num">{fmtMetric(r["f1-score"])}</td><td className="num">{fmtInt(r.support)}</td></tr>; })}</tbody></table></div>
              </Card>) : null}
            {mm.test_curves ? (
              <div className="grid g2">
                <Card title="Precision-recall curve (test)"><CurveChart ariaLabel="Precision-recall curve on the held-out test set" xLabel="Recall" yLabel="Precision" xs={mm.test_curves.pr.recall} ys={mm.test_curves.pr.precision} reference={{ kind: "horizontal", y: mm.test_curves.prevalence }} /><p className="small muted">Dotted line = fraud prevalence in the test split (a no-skill model).</p></Card>
                <Card title="ROC curve (test)"><CurveChart ariaLabel="ROC curve on the held-out test set" xLabel="False positive rate" yLabel="True positive rate" xs={mm.test_curves.roc.fpr} ys={mm.test_curves.roc.tpr} reference={{ kind: "diagonal" }} /><p className="small muted">With rare fraud, ROC-AUC can look high even when precision is poor; prefer the precision-recall view.</p></Card>
              </div>) : <Card><EmptyState title="No curves in this artifact">Curves are written by the training run; older artifacts or runs without both classes in the test split have none.</EmptyState></Card>}
            {mm.feature_importance ? <Card title={`Feature importance (${mm.feature_importance.type.replace(/_/g, " ")})`}><HBarChart ariaLabel="Top feature importances" xLabel={`Importance (${mm.feature_importance.type})`} data={mm.feature_importance.features.map((n, i) => ({ label: n, value: mm.feature_importance!.values[i] ?? 0 }))} /></Card> : null}
            <Card title="Threshold analysis (held-out test range)">
              {table.length ? (<>
                <LineChart ariaLabel="Precision, recall and F1 across thresholds" xLabel="Threshold (ranking score)" yLabel="Metric value" xs={table.map((r) => r.threshold)}
                  series={[{ name: "Precision", color: "#3fb7a6", values: table.map((r) => r.precision) }, { name: "Recall", color: "#f2b84b", values: table.map((r) => r.recall) }, { name: "F1", color: "#6aa8ff", values: table.map((r) => r.f1) }]} />
                <div className="tablewrap"><table><thead><tr><th className="num">Threshold</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Flagged share</th><th className="num">FP</th><th className="num">FN</th></tr></thead>
                  <tbody>{table.map((r) => <tr key={r.threshold}><td className="num">{r.threshold.toFixed(2)}</td><td className="num">{fmtMetric(r.precision)}</td><td className="num">{fmtMetric(r.recall)}</td><td className="num">{fmtMetric(r.f1)}</td><td className="num">{fmtRate(r.flagged_rate)}</td><td className="num">{fmtInt(r.fp)}</td><td className="num">{fmtInt(r.fn)}</td></tr>)}</tbody></table></div>
              </>) : <EmptyState title="No threshold table in this artifact" />}
            </Card>
          </div>
        );
      }}</Async>
    </>
  );
}
