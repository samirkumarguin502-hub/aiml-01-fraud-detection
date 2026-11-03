import { useState, type FormEvent } from "react";
import { useApi } from "../api-context.tsx";
import { PageHead } from "../components/PageHead.tsx";
import { Badge, Banner, Card, ErrorState, Field, ScoreNote } from "../components/ui.tsx";
import { useAsync } from "../hooks/useAsync.ts";
import { fmtScore } from "../lib/format.ts";
import type { Prediction } from "../lib/types.ts";
import { BALANCE_FIELDS, inputFieldsOf, validateTransactionForm, type BalanceField, type TxFields } from "../lib/validate.ts";

const INITIAL: TxFields = { step: "", type: "TRANSFER", amount: "", nameOrig: "", nameDest: "", oldbalanceOrg: "", oldbalanceDest: "", newbalanceOrig: "", newbalanceDest: "" };
const BALANCE_LABEL: Record<BalanceField, string> = { oldbalanceOrg: "Sender balance before", oldbalanceDest: "Receiver balance before", newbalanceOrig: "Sender balance after", newbalanceDest: "Receiver balance after" };

export function Score() {
  const api = useApi();
  const [info] = useAsync((s) => api.model(s), []);  // which inputs does the LOADED model need?
  const fields = inputFieldsOf(info.status === "ready" ? info.data : undefined);
  const balanceFields = BALANCE_FIELDS.filter((b) => fields.includes(b));
  const featureSet = info.status === "ready" ? info.data.model?.feature_set : undefined;
  const [f, setF] = useState<TxFields>(INITIAL);
  const [errors, setErrors] = useState<Partial<Record<keyof TxFields, string>>>({});
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<unknown>(null);
  const bind = (k: keyof TxFields) => ({ id: `tx-${k}`, value: f[k] ?? "", "aria-invalid": errors[k] ? (true as const) : undefined, "aria-describedby": errors[k] ? `tx-${k}-err` : undefined, onChange: (e: { target: { value: string } }) => setF((p) => ({ ...p, [k]: e.target.value })) });

  async function submit(e: FormEvent) {
    e.preventDefault(); setResult(null); setError(null);
    const v = validateTransactionForm(f, fields);
    if (!v.ok) { setErrors(v.errors); return; }
    setErrors({}); setBusy(true);
    try { setResult(await api.predict(v.value)); } catch (err) { setError(err); } finally { setBusy(false); }
  }

  return (
    <>
      <PageHead title="Score a transaction" sub="Send one transaction to the model. The result is stored in the transaction history." />
      {info.status === "ready" && info.data.state !== "ready" ? <Banner tone="warn" title="No usable model is loaded">Scoring will return “service unavailable” until a trained model artifact is loaded.</Banner> : null}
      <div className="grid g2">
        <Card title="Transaction">
          <form onSubmit={submit} noValidate aria-label="Single transaction scoring form">
            <div className="form">
              <Field id="tx-step" label="Step (hour index)" error={errors.step} hint="Whole number, e.g. 12"><input inputMode="numeric" {...bind("step")} /></Field>
              <Field id="tx-type" label="Type" error={errors.type} hint="Types unseen in training are scored as “unknown”"><input maxLength={32} {...bind("type")} /></Field>
              <Field id="tx-amount" label="Amount" error={errors.amount}><input inputMode="decimal" {...bind("amount")} /></Field>
              <Field id="tx-nameOrig" label="Sender account" error={errors.nameOrig}><input maxLength={64} autoComplete="off" {...bind("nameOrig")} /></Field>
              <Field id="tx-nameDest" label="Receiver account" error={errors.nameDest}><input maxLength={64} autoComplete="off" {...bind("nameDest")} /></Field>
              {balanceFields.map((b) => <Field key={b} id={`tx-${b}`} label={BALANCE_LABEL[b]} error={errors[b]} hint={`Required by this model (feature set: ${featureSet ?? "?"})`}><input inputMode="decimal" {...bind(b)} /></Field>)}
            </div>
            <div className="row" style={{ marginTop: "1rem" }}>
              <button className="btn primary" type="submit" disabled={busy}>{busy ? "Scoring…" : "Score transaction"}</button>
              <button className="btn" type="button" onClick={() => { setF(INITIAL); setErrors({}); setResult(null); setError(null); }}>Clear</button>
            </div>
            <p className="small muted">{balanceFields.length ? "This model also uses balances known before the transaction. Post-transaction balances are never accepted unless the model was trained as a diagnostic." : "The loaded model uses no balance columns: they can leak the label in PaySim-style data."}</p>
          </form>
        </Card>
        <div aria-live="polite">
          {error ? <ErrorState error={error} /> : null}
          {result ? (
            <Card title="Result">
              <Banner tone={result.flagged ? "bad" : "good"} title={result.flagged ? "Flagged for review" : "Not flagged"}>Ranking score {fmtScore(result.score)} against threshold {fmtScore(result.threshold)}.</Banner>
              <dl className="kv"><dt>Transaction ID</dt><dd>#{result.transaction_id} <a href="#/transactions">view in explorer</a></dd><dt>Model</dt><dd>{result.model.kind} {result.model.metrics_origin ? <Badge tone={result.model.metrics_origin.toUpperCase().startsWith("SYNTH") ? "warn" : "neutral"}>metrics origin: {result.model.metrics_origin}</Badge> : null}</dd></dl>
              {result.warnings.map((w) => <p key={w} className="small" role="note">⚠ {w}</p>)}
              <p><ScoreNote /></p>
            </Card>
          ) : !error ? <Card title="Result"><p className="muted">No result yet. Submit the form to score a transaction.</p></Card> : null}
        </div>
      </div>
    </>
  );
}
