import assert from "node:assert/strict";
import { test } from "node:test";
import { ResponseShapeError, parseBatch, parseModelInfo, parseMonitoring, parsePage, parsePipeline, parsePrediction, parseReady, parseRetrain, parseSummary } from "./guards.ts";

const tx = { id: 1, created_utc: "2026-01-01T00:00:00+00:00", source: "single", step: 1, type: "T", amount: 5, name_orig: "C1", name_dest: "C2", score: 0.2, threshold: 0.7, flagged: false, case_status: null, feedback_label: null };

test("page accepts valid items and empty lists", () => {
  assert.equal(parsePage({ items: [tx], total: 1, page: 1, page_size: 25 }).items.length, 1);
  assert.equal(parsePage({ items: [], total: 0, page: 1, page_size: 25 }).items.length, 0);
});
test("page rejects bad shapes", () => {
  for (const bad of [null, [], { items: "x", total: 1, page: 1, page_size: 1 }, { items: [{ ...tx, score: "0.2" }], total: 1, page: 1, page_size: 1 }, { items: [{ ...tx, flagged: 1 }], total: 1, page: 1, page_size: 1 }, { items: [], total: NaN, page: 1, page_size: 1 }])
    assert.throws(() => parsePage(bad), ResponseShapeError);
});
test("prediction and batch guards", () => {
  assert.equal(parsePrediction({ transaction_id: 1, score: 0.4, threshold: 0.5, flagged: false, score_note: "n", warnings: [] }).score, 0.4);
  assert.throws(() => parsePrediction({ transaction_id: 1, score: null, threshold: 0.5, flagged: false, score_note: "n", warnings: [] }));
  assert.throws(() => parseBatch({ batch_id: "b", rows: 1, flagged: 0, results: [{ row: 2 }] }));
});
test("model info: unavailable needs no model body; ready requires one", () => {
  assert.equal(parseModelInfo({ state: "unavailable", reason: "none", model: null }).state, "unavailable");
  assert.throws(() => parseModelInfo({ state: "ready", reason: null, model: null }));
  assert.throws(() => parseModelInfo({ state: "weird", reason: null, model: null }));
  const ready = { state: "ready", reason: null, model: { kind: "xgboost", threshold: { threshold: 0.8 }, metrics: { origin: "SYNTHETIC" } } };
  assert.equal(parseModelInfo(ready).state, "ready");
});
test("summary, monitoring, pipeline, retrain and ready guards", () => {
  const s = { empty: true, total_transactions: 0, flagged_transactions: 0, score_histogram: [], by_day: [], by_type: [], case_status_counts: {}, feedback: {}, model: {} };
  assert.equal(parseSummary(s).empty, true);
  assert.throws(() => parseSummary({ ...s, total_transactions: "0" }));
  const m = { status: "insufficient_data", drift_alert: false, alert_reasons: [], volume: {}, data_quality: {}, retraining_note: "n", drift: null, performance: null };
  assert.equal(parseMonitoring(m).status, "insufficient_data");
  assert.throws(() => parseMonitoring({ ...m, status: "fine" }));
  assert.equal(parsePipeline({ optional_components: {}, note: "n", model_state: "ready", database: "ok" }).database, "ok");
  assert.equal(parseRetrain({ id: 1, status: "requested", reason: "r", events: [], allowed_next: [] }).id, 1);
  assert.throws(() => parseRetrain({ id: "1", status: "requested", reason: "r", events: [], allowed_next: [] }));
  assert.equal(parseReady({ ready: false, checks: { database: "ok", model: "invalid" } }).ready, false);
});
