import { useEffect, useRef, useState } from "react";
import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { TxTable } from "../components/TxTable.tsx";
import { Async, Banner, Card, EmptyState, Stat } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtInt, label } from "../lib/format.ts";

export function Replay() {
  const api = useApi();
  const [auto, setAuto] = useState(true);
  const [s, reload] = useAsync((sig) => api.transactions({ page: 1, pageSize: 15, sort: "id", order: "desc" }, sig), []);
  const [p, reloadP] = useAsync((sig) => api.pipeline(sig), []);
  const firstId = useRef<number | null>(null);
  useEffect(() => { if (!auto) return; const t = setInterval(() => { reload(); reloadP(); }, 5000); return () => clearInterval(t); }, [auto, reload, reloadP]);
  return (
    <>
      <PageHead title="Live / replay monitor" sub="Shows what the API has stored, refreshed every 5 seconds." actions={<label style={{ margin: 0 }}><input type="checkbox" style={{ width: "auto" }} checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Auto-refresh</label>} />
      <Banner tone="info" title="This is not a production feed">Rows appear only when something calls the scoring API — typically <span className="mono">python -m streaming api --data &lt;csv&gt;</span> replaying a dataset file, or manual scoring. No Kafka broker is assumed or checked.</Banner>
      <div className="grid">
        <Async state={s} reload={reload}>{(d) => {
          if (firstId.current === null) firstId.current = d.items[0]?.id ?? 0;
          const added = d.items.length ? Math.max(0, (d.items[0]?.id ?? 0) - (firstId.current ?? 0)) : 0;
          return (<>
            <div className="grid g4"><Stat label="Total stored" value={fmtInt(d.total)} /><Stat label="New since opening" value={fmtInt(added)} hint="by transaction id" /></div>
            <Card title="Latest transactions">{d.items.length === 0 ? <EmptyState title="Nothing stored yet">Start a replay or score a transaction.</EmptyState> : <TxTable items={d.items} />}</Card>
          </>);
        }}</Async>
        <Card title="Last replay run (outcome counts only)">
          <Async state={p} reload={reloadP}>{(d) => d.last_replay_summary?.outcomes ? (<>
            <p className="small muted">Mode {d.last_replay_summary.mode}, {d.last_replay_summary.finished ? "finished" : "in progress"}, {d.last_replay_summary.processed} processed. Transaction content is never recorded.</p>
            <div className="row">{Object.entries(d.last_replay_summary.outcomes).map(([k, v]) => <span key={k} className="badge">{label(k)}: {v}</span>)}</div>
          </>) : <p className="muted">No replay summary found.</p>}</Async>
        </Card>
      </div>
    </>
  );
}
