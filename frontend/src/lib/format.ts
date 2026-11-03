// Formatting helpers. Scores are shown as plain numbers, never as percentages or "probability".
export const fmtInt = (n: number) => new Intl.NumberFormat("en-US").format(n);
export const fmtScore = (n: number | null | undefined) => (n === null || n === undefined || !Number.isFinite(n) ? "n/a" : n.toFixed(4));
export const fmtMetric = (n: number | null | undefined) => (n === null || n === undefined || !Number.isFinite(n) ? "n/a" : n.toFixed(4));
export const fmtRate = (n: number | null | undefined) => (n === null || n === undefined || !Number.isFinite(n) ? "n/a" : `${(n * 100).toFixed(2)}%`);
export const fmtAmount = (n: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(n);
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "n/a";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "n/a" : d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}
export const label = (s: string) => s.replace(/_/g, " ");
export const shortHash = (h: string | null | undefined) => (h ? h.slice(0, 10) : "n/a");
