// CSV export with spreadsheet-formula neutralisation (CSV injection).
const DANGEROUS = /^[=+\-@\t\r]/;

export function csvCell(v: unknown): string {
  let s = v === null || v === undefined ? "" : String(v);
  if (typeof v !== "number" && DANGEROUS.test(s)) s = "'" + s;
  if (/[",\n\r]/.test(s)) s = `"${s.replace(/"/g, '""')}"`;
  return s;
}
export function toCsv(rows: Record<string, unknown>[], columns: string[]): string {
  const lines = [columns.map(csvCell).join(",")];
  for (const r of rows) lines.push(columns.map((c) => csvCell(r[c])).join(","));
  return lines.join("\r\n") + "\r\n";
}
export function batchResultsCsv(results: { row: number; transaction_id: number; score: number; flagged: boolean }[]): string {
  return toCsv(results.map((r) => ({ csv_row: r.row, transaction_id: r.transaction_id, ranking_score: r.score, flagged: r.flagged ? "yes" : "no" })), ["csv_row", "transaction_id", "ranking_score", "flagged"]);
}
