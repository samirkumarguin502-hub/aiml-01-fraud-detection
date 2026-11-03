import { useRef, useState } from "react";
import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { ErrorState, Banner, Card, ScoreNote } from "../components/ui.tsx";
import { ApiError } from "../lib/api.ts";
import { batchResultsCsv } from "../lib/csv.ts";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtInt, fmtScore } from "../lib/format.ts";
import type { BatchResult } from "../lib/types.ts";
import { MAX_UPLOAD_BYTES, inputFieldsOf, validateCsvFile } from "../lib/validate.ts";

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}
interface RowIssues { total_invalid_rows?: number; first?: { row: number; fields: string[] }[]; missing?: string[]; required?: string[] }

export function Batch() {
  const api = useApi();
  const [info] = useAsync((s) => api.model(s), []);
  const columns = inputFieldsOf(info.status === "ready" ? info.data : undefined);
  const input = useRef<HTMLInputElement>(null);
  const [fileErr, setFileErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<BatchResult | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function onSubmit(e: { preventDefault(): void }) {
    e.preventDefault(); setRes(null); setError(null);
    const file = input.current?.files?.[0];
    if (!file) { setFileErr("Choose a CSV file first."); return; }
    const bad = validateCsvFile(file); setFileErr(bad); if (bad) return;
    setBusy(true);
    try { setRes(await api.predictBatch(await file.text())); } catch (err) { setError(err); } finally { setBusy(false); }
  }
  const issues = error instanceof ApiError && typeof error.details === "object" && error.details !== null && !Array.isArray(error.details) ? (error.details as RowIssues) : null;

  return (
    <>
      <PageHead title="Batch scoring" sub={`Upload a CSV (max ${Math.round(MAX_UPLOAD_BYTES / 1e6)} MB). Columns required by the loaded model: ${columns.join(", ")}.`} />
      <Card title="Upload">
        <form onSubmit={onSubmit} aria-label="Batch CSV upload">
          <label htmlFor="csv">CSV file</label>
          <input id="csv" ref={input} type="file" accept=".csv,text/csv" aria-invalid={fileErr ? true : undefined} aria-describedby={fileErr ? "csv-err" : undefined} onChange={() => setFileErr(null)} />
          {fileErr ? <div className="err" id="csv-err" role="alert">{fileErr}</div> : <div className="hint">Other columns in the file (including the label and any post-transaction balances) are ignored by the server and never stored.</div>}
          <div className="row" style={{ marginTop: ".8rem" }}><button className="btn primary" type="submit" disabled={busy}>{busy ? "Scoring…" : "Score file"}</button></div>
        </form>
      </Card>
      <div aria-live="polite" style={{ marginTop: "1rem" }}>
        {error ? (<><ErrorState error={error} />
          {issues?.first?.length ? <Card title={`First ${issues.first.length} of ${issues.total_invalid_rows ?? "?"} invalid rows`}><ul className="small">{issues.first.map((r) => <li key={r.row}>Row {r.row}: check {r.fields.join(", ")}</li>)}</ul><p className="small muted">Values are not echoed back for privacy; open the file at those rows.</p></Card> : null}
          {issues?.missing?.length ? <Card><p>Missing columns: <span className="mono">{issues.missing.join(", ")}</span></p></Card> : null}</>) : null}
        {res ? (
          <Card title="Results" actions={<button className="btn" onClick={() => download(`scores-${res.batch_id.slice(0, 8)}.csv`, batchResultsCsv(res.results))}>Download results CSV</button>}>
            <Banner tone="good" title={`${fmtInt(res.rows)} rows scored, ${fmtInt(res.flagged)} flagged`}>{res.unknown_type_rows ? `${res.unknown_type_rows} row(s) had a transaction type unseen in training. ` : ""}Batch {res.batch_id.slice(0, 8)} — all rows are now in the transaction explorer.</Banner>
            <div className="tablewrap"><table><caption className="small muted" style={{ textAlign: "left" }}>First 50 rows shown; the download contains all rows. Formula-like text is neutralised in the CSV.</caption>
              <thead><tr><th>CSV row</th><th>Transaction</th><th className="num">Score</th><th>Result</th></tr></thead>
              <tbody>{res.results.slice(0, 50).map((r) => <tr key={r.transaction_id}><td>{r.row}</td><td>#{r.transaction_id}</td><td className="num">{fmtScore(r.score)}</td><td>{r.flagged ? "flagged" : "not flagged"}</td></tr>)}</tbody></table></div>
            <p><ScoreNote /></p>
          </Card>
        ) : null}
      </div>
    </>
  );
}
