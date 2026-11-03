import { useState } from "react";
import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { CASE_STATUSES, TransactionDetail } from "../components/TransactionDetail.tsx";
import { TxTable } from "../components/TxTable.tsx";
import { Async, Card, EmptyState, Field, Pagination } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { label } from "../lib/format.ts";

interface Filters { flagged: string; type: string; source: string; caseStatus: string; minScore: string; maxScore: string; sort: string; order: "asc" | "desc" }
const EMPTY: Filters = { flagged: "", type: "", source: "", caseStatus: "", minScore: "", maxScore: "", sort: "id", order: "desc" };
const num = (s: string) => (s.trim() === "" || Number.isNaN(Number(s)) ? undefined : Number(s));

export function Transactions() {
  const api = useApi();
  const [draft, setDraft] = useState<Filters>(EMPTY);
  const [f, setF] = useState<Filters>(EMPTY);
  const [page, setPage] = useState(1);
  const [sel, setSel] = useState<number | null>(null);
  const set = (k: keyof Filters) => (e: { target: { value: string } }) => setDraft((d) => ({ ...d, [k]: e.target.value }));
  const minScore = num(f.minScore), maxScore = num(f.maxScore);
  const [s, reload] = useAsync((sig) => api.transactions({
    page, pageSize: 20, sort: f.sort, order: f.order,
    ...(f.flagged ? { flagged: f.flagged === "yes" } : {}), ...(f.type.trim() ? { type: f.type.trim() } : {}), ...(f.source ? { source: f.source } : {}),
    ...(f.caseStatus ? { caseStatus: f.caseStatus } : {}), ...(minScore !== undefined ? { minScore } : {}), ...(maxScore !== undefined ? { maxScore } : {}),
  }, sig), [page, JSON.stringify(f)]);
  return (
    <>
      <PageHead title="Transaction explorer" sub="Every transaction stored by the API, newest first by default." />
      <Card title="Filters">
        <form className="form" onSubmit={(e) => { e.preventDefault(); setF(draft); setPage(1); }}>
          <Field id="f-flag" label="Result"><select id="f-flag" value={draft.flagged} onChange={set("flagged")}><option value="">All</option><option value="yes">Flagged</option><option value="no">Not flagged</option></select></Field>
          <Field id="f-type" label="Type"><input id="f-type" value={draft.type} onChange={set("type")} maxLength={32} placeholder="e.g. TRANSFER" /></Field>
          <Field id="f-src" label="Source"><select id="f-src" value={draft.source} onChange={set("source")}><option value="">All</option><option value="single">Single</option><option value="batch">Batch</option></select></Field>
          <Field id="f-case" label="Case status"><select id="f-case" value={draft.caseStatus} onChange={set("caseStatus")}><option value="">All</option>{CASE_STATUSES.map((c) => <option key={c} value={c}>{label(c)}</option>)}</select></Field>
          <Field id="f-min" label="Min score"><input id="f-min" inputMode="decimal" value={draft.minScore} onChange={set("minScore")} /></Field>
          <Field id="f-max" label="Max score"><input id="f-max" inputMode="decimal" value={draft.maxScore} onChange={set("maxScore")} /></Field>
          <Field id="f-sort" label="Sort by"><select id="f-sort" value={draft.sort} onChange={set("sort")}>{["id", "created_utc", "score", "amount", "step"].map((c) => <option key={c} value={c}>{c}</option>)}</select></Field>
          <Field id="f-ord" label="Order"><select id="f-ord" value={draft.order} onChange={set("order")}><option value="desc">Descending</option><option value="asc">Ascending</option></select></Field>
          <div className="row" style={{ alignItems: "end" }}><button className="btn primary" type="submit">Apply</button><button className="btn" type="button" onClick={() => { setDraft(EMPTY); setF(EMPTY); setPage(1); }}>Reset</button></div>
        </form>
      </Card>
      <div style={{ marginTop: "1rem" }}><Card>
        <Async state={s} reload={reload}>{(d) => d.total === 0 ? (
          <EmptyState title="No transactions match">{JSON.stringify(f) === JSON.stringify(EMPTY) ? "Nothing has been scored yet. Score a transaction or upload a CSV." : "Try widening the filters."}</EmptyState>
        ) : (<><TxTable items={d.items} onSelect={setSel} selectedId={sel} /><Pagination page={d.page} pageSize={d.page_size} total={d.total} onPage={setPage} /></>)}</Async>
      </Card></div>
      {sel !== null ? <div style={{ marginTop: "1rem" }}><TransactionDetail id={sel} onChanged={reload} onClose={() => setSel(null)} /></div> : null}
    </>
  );
}
