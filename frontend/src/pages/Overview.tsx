import { useApi } from "../api-context.tsx";
import { BarChart } from "../components/charts.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { Async, Banner, Card, EmptyState, Stat } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtInt, fmtRate, label } from "../lib/format.ts";

export function Overview() {
  const api = useApi();
  const [s, reload] = useAsync((sig) => api.summary(sig), []);
  return (
    <>
      <PageHead title="Overview" sub="Counts and distributions of transactions scored through this API." actions={<button className="btn" onClick={reload}>Refresh</button>} />
      <Async state={s} reload={reload} label="Loading summary">{(d) => (
        <>
          {d.model.state !== "ready" ? <Banner tone="warn" title="No usable model is loaded">Predictions are disabled. {d.model.reason ?? ""} Anything below was stored earlier; nothing is simulated.</Banner> : null}
          {d.model.metrics_origin?.toUpperCase().startsWith("SYNTH") ? <Banner tone="warn" title="Scores on this dashboard come from a model trained on SYNTHETIC data">Do not read them as real fraud findings.</Banner> : null}
          {d.empty ? (
            <Card><EmptyState title="No transactions scored yet" action={<div className="row" style={{ justifyContent: "center" }}><a className="btn primary" href="#/score">Score a transaction</a><a className="btn" href="#/batch">Upload a CSV</a></div>}>This dashboard only shows transactions that were scored through the API. Nothing is pre-filled.</EmptyState></Card>
          ) : (
            <div className="grid">
              <div className="grid g4">
                <Stat label="Transactions scored" value={fmtInt(d.total_transactions)} />
                <Stat label="Flagged" value={fmtInt(d.flagged_transactions)} hint="score ≥ model threshold" />
                <Stat label="Flag rate" value={fmtRate(d.total_transactions ? d.flagged_transactions / d.total_transactions : null)} hint="share of scored transactions" />
                <Stat label="Open cases" value={fmtInt(d.case_status_counts["open"] ?? 0)} hint={`${fmtInt(d.feedback.total)} analyst label(s)`} />
              </div>
              <div className="grid g2">
                <Card title="Score distribution"><BarChart ariaLabel="Histogram of ranking scores" xLabel="Ranking score (uncalibrated)" yLabel="Transactions" data={d.score_histogram.map((b) => ({ label: b.bin_start.toFixed(1), value: b.count, tip: `${b.bin_start.toFixed(1)}–${b.bin_end.toFixed(1)}: ${b.count}` }))} /></Card>
                <Card title="Volume by day (UTC)">{d.by_day.length ? <BarChart ariaLabel="Transactions scored per day" xLabel="Day" yLabel="Transactions" data={d.by_day.map((x) => ({ label: x.day.slice(5), value: x.transactions, tip: `${x.day}: ${x.transactions} scored, ${x.flagged} flagged` }))} /> : <EmptyState title="No daily data" />}</Card>
              </div>
              <Card title="By transaction type">
                <div className="tablewrap"><table><thead><tr><th>Type</th><th className="num">Scored</th><th className="num">Flagged</th><th className="num">Flag rate</th></tr></thead>
                  <tbody>{d.by_type.map((t) => <tr key={t.type}><td>{t.type}</td><td className="num">{fmtInt(t.transactions)}</td><td className="num">{fmtInt(t.flagged)}</td><td className="num">{fmtRate(t.transactions ? t.flagged / t.transactions : null)}</td></tr>)}</tbody></table></div>
              </Card>
              <Card title="Case workflow">
                <div className="row">{Object.entries(d.case_status_counts).map(([k, v]) => <span key={k} className="badge info">{label(k)}: {v}</span>)}</div>
              </Card>
              <ul className="small muted">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>
            </div>
          )}
        </>
      )}</Async>
    </>
  );
}
