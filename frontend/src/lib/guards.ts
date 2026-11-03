// Runtime validation of API responses. The UI never renders a response that fails these checks.
import type { BatchResult, Health, MonitoringReport, ModelInfo, Page, PipelineStatus, Prediction, Ready, RetrainRequest, Summary, TransactionRow } from "./types.ts";

export class ResponseShapeError extends Error {
  constructor(what: string) { super(`Unexpected response from the API (${what}).`); this.name = "ResponseShapeError"; }
}

type Rec = Record<string, unknown>;
const isRec = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function rec(v: unknown, what: string): Rec { if (!isRec(v)) throw new ResponseShapeError(what); return v; }
function need(r: Rec, key: string, ok: (v: unknown) => boolean, what: string): void {
  if (!ok(r[key])) throw new ResponseShapeError(`${what}.${key}`);
}
const str = (v: unknown) => typeof v === "string";
const bool = (v: unknown) => typeof v === "boolean";
const arr = (v: unknown) => Array.isArray(v);
const rec_ = (v: unknown) => isRec(v);
const strOrNull = (v: unknown) => v === null || typeof v === "string";
const numOrNull = (v: unknown) => v === null || isNum(v);

export function parseHealth(v: unknown): Health {
  const r = rec(v, "health"); need(r, "status", str, "health"); return r as unknown as Health;
}
export function parseReady(v: unknown): Ready {
  const r = rec(v, "ready"); need(r, "ready", bool, "ready"); need(r, "checks", rec_, "ready");
  const c = r["checks"] as Rec; need(c, "database", str, "ready.checks"); need(c, "model", str, "ready.checks");
  return r as unknown as Ready;
}
export function parseTransaction(v: unknown): TransactionRow {
  const r = rec(v, "transaction");
  for (const k of ["id", "step", "amount", "score", "threshold"]) need(r, k, isNum, "transaction");
  for (const k of ["type", "name_orig", "name_dest", "created_utc", "source"]) need(r, k, str, "transaction");
  need(r, "flagged", bool, "transaction"); need(r, "case_status", strOrNull, "transaction");
  need(r, "feedback_label", numOrNull, "transaction");
  return r as unknown as TransactionRow;
}
export function parsePage(v: unknown): Page<TransactionRow> {
  const r = rec(v, "page");
  for (const k of ["total", "page", "page_size"]) need(r, k, isNum, "page");
  need(r, "items", arr, "page");
  (r["items"] as unknown[]).forEach(parseTransaction);
  return r as unknown as Page<TransactionRow>;
}
export function parsePrediction(v: unknown): Prediction {
  const r = rec(v, "prediction");
  for (const k of ["transaction_id", "score", "threshold"]) need(r, k, isNum, "prediction");
  need(r, "flagged", bool, "prediction"); need(r, "score_note", str, "prediction"); need(r, "warnings", arr, "prediction");
  return r as unknown as Prediction;
}
export function parseBatch(v: unknown): BatchResult {
  const r = rec(v, "batch");
  for (const k of ["rows", "flagged"]) need(r, k, isNum, "batch");
  need(r, "batch_id", str, "batch"); need(r, "results", arr, "batch");
  for (const x of r["results"] as unknown[]) { const q = rec(x, "batch.results[]"); need(q, "row", isNum, "batch.results"); need(q, "score", isNum, "batch.results"); need(q, "flagged", bool, "batch.results"); need(q, "transaction_id", isNum, "batch.results"); }
  return r as unknown as BatchResult;
}
export function parseModelInfo(v: unknown): ModelInfo {
  const r = rec(v, "model"); need(r, "state", str, "model");
  if (!["ready", "unavailable", "invalid"].includes(r["state"] as string)) throw new ResponseShapeError("model.state");
  need(r, "reason", strOrNull, "model");
  if (r["state"] === "ready") {
    const m = rec(r["model"], "model.model"); need(m, "kind", str, "model.model"); need(m, "threshold", rec_, "model.model");
    need(m, "metrics", rec_, "model.model"); need(rec(m["metrics"], "metrics"), "origin", str, "model.metrics");
    need(rec(m["threshold"], "threshold"), "threshold", isNum, "model.threshold");
  }
  return r as unknown as ModelInfo;
}
export function parseSummary(v: unknown): Summary {
  const r = rec(v, "summary");
  for (const k of ["total_transactions", "flagged_transactions"]) need(r, k, isNum, "summary");
  need(r, "empty", bool, "summary"); need(r, "score_histogram", arr, "summary"); need(r, "by_day", arr, "summary");
  need(r, "by_type", arr, "summary"); need(r, "case_status_counts", rec_, "summary"); need(r, "model", rec_, "summary");
  need(r, "feedback", rec_, "summary");
  return r as unknown as Summary;
}
export function parseMonitoring(v: unknown): MonitoringReport {
  const r = rec(v, "monitoring");
  need(r, "status", str, "monitoring");
  if (!["ok", "no_model", "insufficient_data", "no_reference"].includes(r["status"] as string)) throw new ResponseShapeError("monitoring.status");
  need(r, "drift_alert", bool, "monitoring"); need(r, "alert_reasons", arr, "monitoring"); need(r, "volume", rec_, "monitoring");
  need(r, "data_quality", rec_, "monitoring"); need(r, "retraining_note", str, "monitoring");
  if (r["drift"] !== null) need(r, "drift", rec_, "monitoring");
  if (r["performance"] !== null) need(r, "performance", rec_, "monitoring");
  return r as unknown as MonitoringReport;
}
export function parsePipeline(v: unknown): PipelineStatus {
  const r = rec(v, "pipeline"); need(r, "optional_components", rec_, "pipeline"); need(r, "note", str, "pipeline");
  need(r, "model_state", str, "pipeline"); need(r, "database", str, "pipeline");
  return r as unknown as PipelineStatus;
}
export function parseRetrain(v: unknown): RetrainRequest {
  const r = rec(v, "retrain"); need(r, "id", isNum, "retrain"); need(r, "status", str, "retrain");
  need(r, "reason", str, "retrain"); need(r, "events", arr, "retrain"); need(r, "allowed_next", arr, "retrain");
  return r as unknown as RetrainRequest;
}
export function parseRetrainList(v: unknown): { items: RetrainRequest[]; note: string } {
  const r = rec(v, "retrain list"); need(r, "items", arr, "retrain list"); need(r, "note", str, "retrain list");
  (r["items"] as unknown[]).forEach(parseRetrain);
  return r as unknown as { items: RetrainRequest[]; note: string };
}
