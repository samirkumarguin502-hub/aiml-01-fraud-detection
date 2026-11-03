import { useState } from "react";
import { useApi } from "../api-context.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { describeError } from "../lib/errors.ts";
import { fmtAmount, fmtDate, fmtScore, label, shortHash } from "../lib/format.ts";
import { validateNote } from "../lib/validate.ts";
import { Async, Badge, Banner, Card, Field, ScoreNote } from "./ui.tsx";

export const CASE_STATUSES = ["open", "in_review", "confirmed_fraud", "false_positive", "closed"] as const;

export function TransactionDetail({ id, onChanged, onClose }: { id: number; onChanged?: () => void; onClose?: () => void }) {
  const api = useApi();
  const [state, reload] = useAsync((s) => api.transaction(id, s), [id]);
  const [status, setStatus] = useState<string>("in_review");
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<{ tone: "good" | "bad"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(action: () => Promise<unknown>, ok: string) {
    setBusy(true); setMsg(null);
    try { await action(); setMsg({ tone: "good", text: ok }); reload(); onChanged?.(); }
    catch (e) { setMsg({ tone: "bad", text: describeError(e).message }); }
    finally { setBusy(false); }
  }
  const noteErr = validateNote(note);

  return (
    <Card title={`Transaction #${id}`} actions={onClose ? <button className="btn" onClick={onClose}>Close</button> : undefined}>
      <Async state={state} reload={reload}>{(t) => (
        <div className="grid g2">
          <div>
            <dl className="kv">
              <dt>Type</dt><dd>{t.type}</dd><dt>Amount</dt><dd>{fmtAmount(t.amount)}</dd><dt>Step</dt><dd>{t.step}</dd>
              <dt>Sender</dt><dd className="mono">{t.name_orig}</dd><dt>Receiver</dt><dd className="mono">{t.name_dest}</dd>
              <dt>Ranking score</dt><dd>{fmtScore(t.score)} <span className="muted">(threshold {fmtScore(t.threshold)})</span></dd>
              <dt>Result</dt><dd>{t.flagged ? <Badge tone="bad">flagged</Badge> : <Badge>not flagged</Badge>}</dd>
              <dt>Scored</dt><dd>{fmtDate(t.created_utc)} via {t.source}</dd>
              <dt>Model</dt><dd>{t.model_kind} <span className="mono">{shortHash(t.model_sha256)}</span>{t.model_data_label ? <> · metrics origin: {t.model_data_label}</> : null}</dd>
              {!t.type_in_training_vocab ? <><dt>Note</dt><dd>Transaction type was not seen in training (scored in the “unknown” bucket).</dd></> : null}
            </dl>
            <p><ScoreNote /></p>
            {t.case_history?.length ? <><h3>Case history</h3><ul className="small">{t.case_history.map((e, i) => <li key={i}>{fmtDate(e.created_utc)}: {e.from_status ? label(e.from_status) : "created"} → {label(e.to_status)}</li>)}</ul></> : null}
          </div>
          <div>
            <h3>Review</h3>
            <p className="small muted">Current case status: {t.case_status ? label(t.case_status) : "no case"}{t.case_note ? ` — “${t.case_note}”` : ""}</p>
            <div className="grid">
              <Field id={`status-${id}`} label="Case status"><select id={`status-${id}`} value={status} onChange={(e) => setStatus(e.target.value)}>{CASE_STATUSES.map((s) => <option key={s} value={s}>{label(s)}</option>)}</select></Field>
              <Field id={`note-${id}`} label="Note (optional)" error={noteErr ?? undefined}><textarea id={`note-${id}`} rows={2} value={note} onChange={(e) => setNote(e.target.value)} aria-invalid={noteErr ? true : undefined} /></Field>
              <div><button className="btn primary" disabled={busy || !!noteErr} onClick={() => run(() => api.updateCase(id, status, note.trim() || undefined), "Case updated.")}>Save case status</button></div>
            </div>
            <h3 style={{ marginTop: "1rem" }}>Analyst label {t.feedback_label !== null ? <Badge tone="info">{t.feedback_label === 1 ? "fraud" : "not fraud"}</Badge> : <span className="muted small">(none yet)</span>}</h3>
            <div className="row">
              <button className="btn danger" disabled={busy} onClick={() => run(() => api.setFeedback(id, 1), "Label saved: fraud.")}>Confirm fraud</button>
              <button className="btn" disabled={busy} onClick={() => run(() => api.setFeedback(id, 0), "Label saved: not fraud.")}>Mark not fraud</button>
            </div>
            <p className="small muted">Labels feed the labelled-subset performance view; they never retrain the model automatically.</p>
            <div aria-live="polite">{msg ? <Banner tone={msg.tone} title={msg.tone === "good" ? "Saved" : "Could not save"}>{msg.text}</Banner> : null}</div>
          </div>
        </div>
      )}</Async>
    </Card>
  );
}
