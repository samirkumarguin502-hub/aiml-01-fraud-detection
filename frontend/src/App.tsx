import { useEffect, useState, type ReactNode } from "react";
import { useApi } from "./api-context.tsx";
import { Badge } from "./components/ui.tsx";
import { useAsync } from "./hooks/useAsync.ts";
import { About } from "./pages/About.tsx";
import { Alerts } from "./pages/Alerts.tsx";
import { Batch } from "./pages/Batch.tsx";
import { Health } from "./pages/Health.tsx";
import { Model } from "./pages/Model.tsx";
import { Monitoring } from "./pages/Monitoring.tsx";
import { Overview } from "./pages/Overview.tsx";
import { Replay } from "./pages/Replay.tsx";
import { Score } from "./pages/Score.tsx";
import { Transactions } from "./pages/Transactions.tsx";

export const ROUTES: { path: string; title: string; group: string; el: () => ReactNode }[] = [
  { path: "/", title: "Overview", group: "Operations", el: () => <Overview /> },
  { path: "/alerts", title: "Alerts & cases", group: "Operations", el: () => <Alerts /> },
  { path: "/transactions", title: "Transactions", group: "Operations", el: () => <Transactions /> },
  { path: "/replay", title: "Live / replay", group: "Operations", el: () => <Replay /> },
  { path: "/score", title: "Score one", group: "Scoring", el: () => <Score /> },
  { path: "/batch", title: "Batch CSV", group: "Scoring", el: () => <Batch /> },
  { path: "/model", title: "Model & thresholds", group: "Model", el: () => <Model /> },
  { path: "/monitoring", title: "Drift & monitoring", group: "Model", el: () => <Monitoring /> },
  { path: "/health", title: "System health", group: "Model", el: () => <Health /> },
  { path: "/about", title: "About & limits", group: "Project", el: () => <About /> },
];

export const routeFromHash = (hash: string) => { const p = hash.replace(/^#/, "") || "/"; return ROUTES.some((r) => r.path === p) ? p : "/"; };

function useHashRoute() {
  const [route, setRoute] = useState(() => routeFromHash(window.location.hash));
  useEffect(() => { const on = () => setRoute(routeFromHash(window.location.hash)); window.addEventListener("hashchange", on); return () => window.removeEventListener("hashchange", on); }, []);
  return route;
}

function StatusPills() {
  const api = useApi();
  const [r, reload] = useAsync((s) => api.ready(s), []);
  useEffect(() => { const t = setInterval(reload, 30000); return () => clearInterval(t); }, [reload]);
  if (r.status === "loading") return <div className="pills"><Badge>checking API…</Badge></div>;
  if (r.status === "error") return <div className="pills"><Badge tone="bad">API offline</Badge></div>;
  return <div className="pills"><Badge tone="good">API online</Badge><Badge tone={r.data.checks.model === "ready" ? "good" : "warn"}>model: {r.data.checks.model}</Badge></div>;
}

export function App() {
  const route = useHashRoute();
  const current = ROUTES.find((r) => r.path === route) ?? ROUTES[0]!;
  useEffect(() => { document.title = `${current.title} · AIML-01 Fraud Operations`; }, [current.title]);
  const groups = [...new Set(ROUTES.map((r) => r.group))];
  return (
    <div className="shell">
      <a className="skip" href="#main">Skip to content</a>
      <aside className="side">
        <div className="brand"><span className="logo" aria-hidden="true" /> Fraud Operations</div>
        <nav className="nav" aria-label="Main">
          {groups.map((g) => (<div key={g}><div className="group">{g}</div>{ROUTES.filter((r) => r.group === g).map((r) => <a key={r.path} href={`#${r.path}`} {...(r.path === route ? { "aria-current": "page" as const } : {})}>{r.title}</a>)}</div>))}
        </nav>
      </aside>
      <main id="main" className="main" tabIndex={-1}>
        <div className="topbar" style={{ justifyContent: "flex-end" }}><StatusPills /></div>
        {current.el()}
      </main>
    </div>
  );
}
