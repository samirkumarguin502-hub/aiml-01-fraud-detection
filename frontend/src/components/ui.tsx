import type { ReactNode } from "react";
import type { AsyncState } from "../hooks/useAsync.ts";
import { describeError } from "../lib/errors.ts";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info";

export const Card = ({ title, actions, children }: { title?: string; actions?: ReactNode; children: ReactNode }) => (
  <section className="card">{title || actions ? <header>{title ? <h2>{title}</h2> : <span />}{actions}</header> : null}{children}</section>
);
export const Stat = ({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) => (
  <div className="card stat"><div className="l">{label}</div><div className="v">{value}</div>{hint ? <div className="small muted">{hint}</div> : null}</div>
);
export const Badge = ({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) => (
  <span className={`badge ${tone === "neutral" ? "" : tone}`}>{children}</span>
);
export const Banner = ({ tone = "info", title, children }: { tone?: Exclude<Tone, "neutral">; title: string; children?: ReactNode }) => (
  <div className={`banner ${tone}`} role={tone === "bad" ? "alert" : "status"}><strong>{title}</strong>{children}</div>
);
export const Loading = ({ label = "Loading" }: { label?: string }) => (
  <div role="status" aria-live="polite" className="empty"><span className="spin" aria-hidden="true" /> <span>{label}…</span></div>
);
export const EmptyState = ({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) => (
  <div className="empty"><h3>{title}</h3>{children ? <p>{children}</p> : null}{action}</div>
);
export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const v = describeError(error);
  return (
    <div className={`banner ${v.kind === "offline" ? "warn" : "bad"}`} role="alert">
      <strong>{v.title}</strong>
      <div>{v.message}</div>
      {v.code || v.requestId ? <div className="small muted mono">{v.code ? `code: ${v.code}` : ""} {v.requestId ? `request: ${v.requestId}` : ""}</div> : null}
      {v.kind === "offline" ? <div className="small muted">Start the API (see docs/api.md) or check the proxy settings. Nothing on this page is made up while the API is offline.</div> : null}
      {onRetry ? <div style={{ marginTop: ".5rem" }}><button className="btn" onClick={onRetry}>Retry</button></div> : null}
    </div>
  );
}
export function Async<T>({ state, reload, children, label }: { state: AsyncState<T>; reload?: () => void; children: (data: T) => ReactNode; label?: string }) {
  if (state.status === "loading") return <Loading {...(label ? { label } : {})} />;
  if (state.status === "error") return <ErrorState error={state.error} {...(reload ? { onRetry: reload } : {})} />;
  return <>{children(state.data)}</>;
}
export const Field = ({ id, label, error, hint, children }: { id: string; label: string; error?: string | undefined; hint?: string; children: ReactNode }) => (
  <div><label htmlFor={id}>{label}</label>{children}{error ? <div className="err" id={`${id}-err`} role="alert">{error}</div> : hint ? <div className="hint">{hint}</div> : null}</div>
);
export function Pagination({ page, pageSize, total, onPage }: { page: number; pageSize: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <nav className="row" aria-label="Pagination" style={{ justifyContent: "space-between", marginTop: ".7rem" }}>
      <span className="small muted">{total === 0 ? "0 results" : `Page ${page} of ${pages} · ${total} result${total === 1 ? "" : "s"}`}</span>
      <span className="row">
        <button className="btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
        <button className="btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
      </span>
    </nav>
  );
}
export const SyntheticBanner = ({ origin }: { origin: string }) => (
  <Banner tone="warn" title="Metrics come from SYNTHETIC test data">
    These numbers describe the model's behaviour on the generated fixture ({origin}). They say nothing about real fraud or the PaySim dataset.
  </Banner>
);
export const ScoreNote = () => <span className="small muted">Ranking score from a class-weighted model — not a calibrated probability of fraud.</span>;
