import { API_BASE_URL, useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { Async, Badge, Banner, Card } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtDate } from "../lib/format.ts";

export function Health() {
  const api = useApi();
  const [h, rh] = useAsync((s) => api.health(s), []);
  const [r, rr] = useAsync((s) => api.ready(s), []);
  const [p, rp] = useAsync((s) => api.pipeline(s), []);
  const refresh = () => { rh(); rr(); rp(); };
  return (
    <>
      <PageHead title="Model, API & pipeline health" sub={`Frontend talks to ${API_BASE_URL}`} actions={<button className="btn" onClick={refresh}>Refresh</button>} />
      <div className="grid g2">
        <Card title="API liveness"><Async state={h} reload={rh}>{(d) => <p><Badge tone="good">{d.status}</Badge> <span className="small muted">The process is up. Liveness says nothing about readiness.</span></p>}</Async></Card>
        <Card title="Readiness"><Async state={r} reload={rr}>{(d) => (<>
          <p><Badge tone={d.ready ? "good" : "warn"}>{d.ready ? "ready" : "not ready"}</Badge></p>
          <dl className="kv"><dt>Database</dt><dd><Badge tone={d.checks.database === "ok" ? "good" : "bad"}>{d.checks.database}</Badge></dd><dt>Model</dt><dd><Badge tone={d.checks.model === "ready" ? "good" : "warn"}>{d.checks.model}</Badge></dd></dl>
          {d.model_reason ? <p className="small muted">{d.model_reason}</p> : null}
        </>)}</Async></Card>
      </div>
      <div style={{ marginTop: "1rem" }}><Card title="Optional pipeline components">
        <Async state={p} reload={rp}>{(d) => (<>
          <Banner tone="info" title="What “installed” means">{d.note}</Banner>
          <div className="tablewrap"><table><thead><tr><th>Component</th><th>Python library</th></tr></thead>
            <tbody>{Object.entries(d.optional_components).map(([k, v]) => <tr key={k}><td>{k}</td><td><Badge tone={v.library_installed ? "info" : "neutral"}>{v.library_installed ? "importable" : "not installed"}</Badge></td></tr>)}</tbody></table></div>
          <p className="small muted">Broker, feature-store and MLflow-server status are not probed by the dashboard, so none is reported as running.</p>
          <h3>Last replay run</h3>
          {d.last_replay_summary ? (<dl className="kv"><dt>Mode</dt><dd>{d.last_replay_summary.mode ?? "n/a"}</dd><dt>Started</dt><dd>{fmtDate(d.last_replay_summary.started_utc)}</dd><dt>Processed</dt><dd>{d.last_replay_summary.processed ?? "n/a"}</dd></dl>) : <p className="muted">No replay has been recorded.</p>}
        </>)}</Async>
      </Card></div>
    </>
  );
}
