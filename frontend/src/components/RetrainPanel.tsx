import { useState } from "react";
import { useApi } from "../api-context.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { describeError } from "../lib/errors.ts";
import { fmtDate, fmtScore, label } from "../lib/format.ts";
import { Async, Badge, Banner, Card, EmptyState, Field } from "./ui.tsx";

const TONE: Record<string, "neutral" | "good" | "warn" | "bad" | "info"> = { requested: "info", candidate_trained: "info", evaluated_passed: "good", evaluated_failed: "bad", promoted: "good", rejected: "bad", cancelled: "neutral" };
// What the API's "allowed_next" statuses mean as UI actions. evaluated_* are decided by the server, never by the client.
const ACTION: Record<string, { to: string; text: string }> = { candidate_trained: { to: "candidate_trained", text: "Register trained candidate" }, evaluated_passed: { to: "evaluate", text: "Run evaluation gate" }, evaluated_failed: { to: "evaluate", text: "Run evaluation gate" }, promoted: { to: "promoted", text: "Record promotion decision" }, rejected: { to: "rejected", text: "Reject" }, cancelled: { to: "cancelled", text: "Cancel" } };

export function RetrainPanel() {
  const api = useApi();
  const [s, reload] = useAsync((sig) => api.retrainList(sig), []);
  const [reason, setReason] = useState("");
  const [candidate, setCandidate] = useState("");
  const [msg, setMsg] = useState<{ tone: "good" | "bad"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  async function run(fn: () => Promise<unknown>, ok: string) {
    setBusy(true); setMsg(null);
    try { await fn(); setMsg({ tone: "good", text: ok }); setReason(""); reload(); } catch (e) { setMsg({ tone: "bad", text: describeError(e).message }); } finally { setBusy(false); }
  }
  return (
    <Card title="Retraining workflow (controlled, audited)">
      <p className="small muted">Stages: requested → candidate trained → evaluated by the server → promotion decision recorded. Promotion here is a recorded decision only; activating a model means pointing <span className="mono">MODEL_ARTIFACT_DIR</span> at it and restarting the API. A drift alert never starts this by itself. There is no authentication yet, so treat this panel as a local-demo control.</p>
      <form className="row" style={{ alignItems: "end" }} onSubmit={(e) => { e.preventDefault(); if (reason.trim()) void run(() => api.retrainCreate(reason.trim()), "Retraining request recorded."); }}>
        <div style={{ flex: "1 1 280px" }}><Field id="rt-reason" label="Reason for a retraining request"><input id="rt-reason" value={reason} maxLength={500} onChange={(e) => setReason(e.target.value)} /></Field></div>
        <button className="btn primary" type="submit" disabled={busy || !reason.trim()}>Request retraining</button>
      </form>
      <div aria-live="polite">{msg ? <Banner tone={msg.tone} title={msg.tone === "good" ? "Done" : "Not allowed"}>{msg.text}</Banner> : null}</div>
      <Async state={s} reload={reload}>{(d) => d.items.length === 0 ? <EmptyState title="No retraining requests">None have been recorded.</EmptyState> : (
        <div className="grid">{d.items.map((r) => (
          <div key={r.id} className="card" style={{ background: "var(--panel-2)" }}>
            <div className="row" style={{ justifyContent: "space-between" }}><strong>#{r.id} · {r.reason}</strong><Badge tone={TONE[r.status] ?? "neutral"}>{label(r.status)}</Badge></div>
            <div className="small muted">Updated {fmtDate(r.updated_utc)}{r.candidate ? ` · candidate ${r.candidate}` : ""}</div>
            {r.evaluation ? <div className="small" style={{ marginTop: ".4rem" }}>Gate: {r.evaluation.passed ? "passed" : "failed"} — candidate test PR-AUC {fmtScore(r.evaluation.candidate_test_pr_auc)} vs active {fmtScore(r.evaluation.active_test_pr_auc)} ({r.evaluation.candidate_origin ?? "?"}). <span className="muted">{r.evaluation.caveat}</span></div> : null}
            {r.allowed_next.length ? (
              <div className="row" style={{ marginTop: ".5rem", alignItems: "end" }}>
                {r.status === "requested" ? <div style={{ flex: "1 1 200px" }}><Field id={`cand-${r.id}`} label="Candidate run folder (name under ml/artifacts)"><input id={`cand-${r.id}`} value={candidate} onChange={(e) => setCandidate(e.target.value)} maxLength={80} /></Field></div> : null}
                {r.allowed_next.map((n) => { const a = ACTION[n]; if (!a || (n === "evaluated_failed")) return null; return (
                  <button key={n} className={`btn ${n === "rejected" || n === "cancelled" ? "danger" : ""}`} disabled={busy || (n === "candidate_trained" && !candidate.trim())}
                    onClick={() => { if (n === "promoted" && !window.confirm("Record a promotion decision? This does not swap the running model.")) return; void run(() => api.retrainTransition(r.id, a.to, undefined, n === "candidate_trained" ? candidate.trim() : undefined), `Request #${r.id} updated.`); }}>{a.text}</button>); })}
              </div>) : <div className="small muted">Final state.</div>}
          </div>))}</div>)}</Async>
    </Card>
  );
}
