import type { ReactNode } from "react";
export const PageHead = ({ title, sub, actions }: { title: string; sub?: string; actions?: ReactNode }) => (
  <div className="topbar"><div><h1>{title}</h1>{sub ? <div className="muted">{sub}</div> : null}</div>{actions}</div>
);
