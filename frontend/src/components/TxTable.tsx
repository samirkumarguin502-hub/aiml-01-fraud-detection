import type { TransactionRow } from "../lib/types.ts";
import { fmtAmount, fmtDate, fmtScore, label } from "../lib/format.ts";
import { Badge } from "./ui.tsx";

export function TxTable({ items, onSelect, selectedId }: { items: TransactionRow[]; onSelect?: (id: number) => void; selectedId?: number | null }) {
  return (
    <div className="tablewrap">
      <table>
        <caption className="small muted" style={{ textAlign: "left", paddingBottom: ".4rem" }}>Scored transactions. “Flagged” means the ranking score reached the model’s threshold.</caption>
        <thead><tr><th>ID</th><th>Time</th><th>Type</th><th className="num">Amount</th><th>Sender</th><th>Receiver</th><th className="num">Score</th><th>Status</th><th>Case</th></tr></thead>
        <tbody>
          {items.map((t) => (
            <tr key={t.id} className={`${onSelect ? "click" : ""} ${selectedId === t.id ? "sel" : ""}`} tabIndex={onSelect ? 0 : undefined}
              onClick={() => onSelect?.(t.id)} onKeyDown={(e) => { if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect(t.id); } }}>
              <td>{t.id}</td><td>{fmtDate(t.created_utc)}</td><td>{t.type}</td><td className="num">{fmtAmount(t.amount)}</td>
              <td className="mono">{t.name_orig}</td><td className="mono">{t.name_dest}</td><td className="num">{fmtScore(t.score)}</td>
              <td>{t.flagged ? <Badge tone="bad">flagged</Badge> : <Badge>not flagged</Badge>}</td>
              <td>{t.case_status ? <Badge tone="info">{label(t.case_status)}</Badge> : <span className="muted">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
