import type { Client } from "./api.ts";
import { parseBatch, parseHealth, parseModelInfo, parseMonitoring, parsePage, parsePipeline, parsePrediction, parseReady, parseRetrain, parseRetrainList, parseSummary, parseTransaction } from "./guards.ts";
import type { TxInput } from "./validate.ts";

export interface ListParams { page?: number; pageSize?: number; sort?: string; order?: "asc" | "desc"; flagged?: boolean; type?: string; source?: string; caseStatus?: string; minScore?: number; maxScore?: number }
type Sig = AbortSignal | undefined;
const sig = (s: Sig) => (s ? { signal: s } : {});

export function makeApi(c: Client) {
  const q = (p: Record<string, string | number | boolean | undefined>) => {
    const s = new URLSearchParams();
    for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== "") s.set(k, String(v));
    const t = s.toString(); return t ? `?${t}` : "";
  };
  const listQuery = (p: ListParams) => q({ page: p.page, page_size: p.pageSize, sort: p.sort, order: p.order, flagged: p.flagged, type: p.type, source: p.source, case_status: p.caseStatus, min_score: p.minScore, max_score: p.maxScore });
  const raw = (v: unknown) => v;
  return {
    health: (s?: Sig) => c.request("/health", parseHealth, sig(s)),
    ready: (s?: Sig) => c.request("/ready", parseReady, { allowStatus: [503], ...sig(s) }),
    model: (s?: Sig) => c.request("/model", parseModelInfo, sig(s)),
    summary: (s?: Sig) => c.request("/dashboard/summary", parseSummary, sig(s)),
    transactions: (p: ListParams, s?: Sig) => c.request(`/transactions${listQuery(p)}`, parsePage, sig(s)),
    alerts: (p: { page?: number; pageSize?: number; caseStatus?: string }, s?: Sig) => c.request(`/alerts${q({ page: p.page, page_size: p.pageSize, case_status: p.caseStatus })}`, parsePage, sig(s)),
    transaction: (id: number, s?: Sig) => c.request(`/transactions/${id}`, parseTransaction, sig(s)),
    predict: (tx: TxInput) => c.request("/predict", parsePrediction, { method: "POST", json: tx }),
    predictBatch: (csvText: string) => c.request("/predict/batch", parseBatch, { method: "POST", body: csvText, contentType: "text/csv" }),
    updateCase: (id: number, status: string, note?: string) => c.request(`/transactions/${id}/case`, raw, { method: "PATCH", json: { status, ...(note ? { note } : {}) } }),
    setFeedback: (id: number, label: 0 | 1, comment?: string) => c.request(`/transactions/${id}/feedback`, raw, { method: "PUT", json: { label, ...(comment ? { comment } : {}) } }),
    monitoring: (s?: Sig) => c.request("/monitoring/status", parseMonitoring, sig(s)),
    pipeline: (s?: Sig) => c.request("/pipeline/status", parsePipeline, sig(s)),
    retrainList: (s?: Sig) => c.request("/monitoring/retraining", parseRetrainList, sig(s)),
    retrainCreate: (reason: string, note?: string) => c.request("/monitoring/retraining", parseRetrain, { method: "POST", json: { reason, ...(note ? { note } : {}) } }),
    retrainTransition: (id: number, to: string, note?: string, candidate?: string) => c.request(`/monitoring/retraining/${id}/transition`, parseRetrain, { method: "POST", json: { to, ...(note ? { note } : {}), ...(candidate ? { candidate } : {}) } }),
  };
}
export type Api = ReturnType<typeof makeApi>;
