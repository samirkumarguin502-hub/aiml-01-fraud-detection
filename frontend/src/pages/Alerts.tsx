import { useState } from "react";
import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { TransactionDetail, CASE_STATUSES } from "../components/TransactionDetail.tsx";
import { TxTable } from "../components/TxTable.tsx";
import { Async, Card, EmptyState, Field, Pagination } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { label } from "../lib/format.ts";

export function Alerts() {
  const api = useApi();
  const [page, setPage] = useState(1);
  const [caseStatus, setCaseStatus] = useState("");
  const [sel, setSel] = useState<number | null>(null);
  const [s, reload] = useAsync((sig) => api.alerts({ page, pageSize: 15, ...(caseStatus ? { caseStatus } : {}) }, sig), [page, caseStatus]);
  return (
    <>
      <PageHead title="Fraud alerts & cases" sub="Transactions whose ranking score reached the model's threshold. Select one to review it." />
      <Card actions={<div style={{ minWidth: 200 }}><Field id="cs" label="Case status"><select id="cs" value={caseStatus} onChange={(e) => { setCaseStatus(e.target.value); setPage(1); }}><option value="">All</option>{CASE_STATUSES.map((c) => <option key={c} value={c}>{label(c)}</option>)}</select></Field></div>}>
        <Async state={s} reload={reload}>{(d) => d.items.length === 0 ? (
          <EmptyState title={caseStatus ? "No alerts with this status" : "No alerts"}>Alerts appear here when a scored transaction reaches the threshold. {caseStatus ? "Try clearing the filter." : "None have been raised yet."}</EmptyState>
        ) : (<><TxTable items={d.items} onSelect={setSel} selectedId={sel} /><Pagination page={d.page} pageSize={d.page_size} total={d.total} onPage={setPage} /></>)}</Async>
      </Card>
      {sel !== null ? <div style={{ marginTop: "1rem" }}><TransactionDetail id={sel} onChanged={reload} onClose={() => setSel(null)} /></div> : null}
    </>
  );
}
