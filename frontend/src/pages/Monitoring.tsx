import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { RetrainPanel } from "../components/RetrainPanel.tsx";
import { Async, Badge, Banner, Card, EmptyState, Stat } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtDate, fmtInt, fmtMetric, fmtRate, label } from "../lib/format.ts";
import type { MonitoringReport } from "../lib/types.ts";

const TONE = { stable: "good", warning: "warn", alert: "bad" } as const;

function Status({ r }: { r: MonitoringReport }) {
  if (r.status === "no_model") return <Banner tone="warn" title="No model loaded">{r.reason} Only volume figures below are available.</Banner>;
  if (r.status === "no_reference") return <Banner tone="warn" title="No reference profile in this model artifact">{r.drift?.reason}</Banner>;
  if (r.status === "insufficient_data") return <Banner tone="info" title="Insufficient data for drift analysis">{r.drift?.reason} No drift numbers are shown until there is enough real data.</Banner>;
  return r.drift_alert ? <Banner tone="bad" title="Drift alert">{r.alert_reasons.join("; ")}. This is a signal to investigate, not an automatic action.</Banner> : <Banner tone="good" title="No significant drift detected">Compared {r.drift?.rows_compared} recent scored transactions with the model's training reference.</Banner>;
}

export function Monitoring() {
  const api = useApi();
  const [s, reload] = useAsync((sig) => api.monitoring(sig), []);
  return (
    <>
      <PageHead title="Data drift & monitoring" sub="Computed from transactions stored by this API, against the reference saved with the active model." actions={<button className="btn" onClick={reload}>Refresh</button>} />
      <Async state={s} reload={reload}>{(r) => (
        <div className="grid">
          <Status r={r} />
          <div className="grid g4">
            <Stat label="Scored transactions" value={fmtInt(r.volume.total_transactions)} />
            <Stat label="Alerts raised" value={fmtInt(r.volume.flagged_transactions)} hint={`flag rate ${fmtRate(r.volume.flag_rate)}`} />
            <Stat label="Rejected requests" value={fmtInt(r.data_quality.rejected_requests_total)} hint="invalid input, by error code" />
            <Stat label="Unseen types" value={r.data_quality.unknown_type_rate === null ? "n/a" : fmtRate(r.data_quality.unknown_type_rate)} hint={`${fmtInt(r.data_quality.unknown_type_rows)} row(s)`} />
          </div>
          <Card title="Input data quality">
            {Object.keys(r.data_quality.rejected_requests_by_code).length === 0 ? <p className="muted">No rejected requests recorded.</p> : <div className="row">{Object.entries(r.data_quality.rejected_requests_by_code).map(([k, v]) => <Badge key={k} tone="warn">{label(k)}: {v}</Badge>)}</div>}
            <p className="small muted">{r.data_quality.note}</p>
          </Card>
          <Card title="Feature distribution drift (PSI)">
            {r.drift?.features ? (<>
              {r.drift.features.map((f) => (<div className="psi" key={f.feature}><span className="mono">{f.feature}</span><span className="bar" aria-hidden="true"><i style={{ width: `${Math.min(100, (f.psi / 0.5) * 100)}%` }} /></span><span>{fmtMetric(f.psi)}</span><Badge tone={TONE[f.level]}>{f.level}</Badge></div>))}
              {r.drift.score ? (<div className="psi" style={{ borderTop: "1px solid var(--line)", marginTop: ".4rem", paddingTop: ".6rem" }}><strong>Score distribution</strong><span className="bar" aria-hidden="true"><i style={{ width: `${Math.min(100, (r.drift.score.psi / 0.5) * 100)}%` }} /></span><span>{fmtMetric(r.drift.score.psi)}</span><Badge tone={TONE[r.drift.score.level]}>{r.drift.score.level}</Badge></div>) : null}
              <p className="small muted">{r.drift.threshold_note} Warning above {r.drift.thresholds.warning}, alert above {r.drift.thresholds.alert}. Score reference: {r.drift.score?.reference}. <span className="mono">day_index</span> is excluded because it grows with time by design.</p>
            </>) : <EmptyState title="No drift numbers" >{r.status === "insufficient_data" ? `Need ${r.drift?.min_rows ?? "more"} scored transactions (have ${r.drift?.rows_compared ?? 0}).` : "Drift is only computed when a model with a reference profile is loaded and enough data exists."}</EmptyState>}
          </Card>
          <Card title="Performance on analyst-labelled transactions">
            {r.performance?.status === "ok" ? (<div className="row"><Stat label="Labelled" value={fmtInt(r.performance.labelled)} /><Stat label="Precision" value={fmtMetric(r.performance.precision)} /><Stat label="Recall" value={fmtMetric(r.performance.recall)} /><Stat label="F1" value={fmtMetric(r.performance.f1)} /></div>)
              : <EmptyState title="Not enough labels">{r.performance?.reason ?? "Label transactions from the alerts page to enable this view."}</EmptyState>}
            {r.performance ? <p className="small muted">{r.performance.warning}</p> : null}
          </Card>
          <Card title="Optional Evidently report"><p className="small muted">The dashboard uses the platform's own PSI monitoring. An Evidently HTML report can be produced offline with <span className="mono">python -m monitoring evidently</span> when Evidently is installed; it is not generated here. Report generated {fmtDate(r.generated_utc)}.</p></Card>
          <Banner tone="info" title="Drift never swaps the model">{r.retraining_note}</Banner>
          <RetrainPanel />
        </div>
      )}</Async>
    </>
  );
}
