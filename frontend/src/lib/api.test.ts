import assert from "node:assert/strict";
import { test } from "node:test";
import { ApiError, NetworkError, createClient } from "./api.ts";
import { makeApi } from "./endpoints.ts";

type Call = { url: string; init: RequestInit };
function fakeFetch(status: number, body: unknown, calls: Call[] = [], text?: string): typeof fetch {
  return (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    return new Response(text ?? JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
}
const ident = (v: unknown) => v;

test("GET success returns parsed data and sends no credentials", async () => {
  const calls: Call[] = [];
  const c = createClient({ baseUrl: "/api/v1/", fetchImpl: fakeFetch(200, { status: "alive" }, calls) });
  assert.deepEqual(await c.request("/health", ident), { status: "alive" });
  assert.equal(calls[0]!.url, "/api/v1/health");
  assert.equal(calls[0]!.init.credentials, "omit");
  const h = calls[0]!.init.headers as Record<string, string>;
  assert.ok(!Object.keys(h).some((k) => k.toLowerCase() === "authorization"));
});

test("error body becomes ApiError with code, message and request id; no input echo needed", async () => {
  const body = { error: { code: "model_unavailable", message: "No usable trained model is loaded.", request_id: "abc123", details: { model_state: "unavailable" } } };
  const c = createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(503, body) });
  await assert.rejects(c.request("/predict", ident, { method: "POST", json: {} }), (e: unknown) => {
    assert.ok(e instanceof ApiError);
    assert.equal(e.status, 503); assert.equal(e.code, "model_unavailable"); assert.equal(e.requestId, "abc123");
    return true;
  });
});

test("non-JSON error pages and empty bodies are handled safely", async () => {
  const c = createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(502, null, [], "<html>Bad gateway</html>") });
  await assert.rejects(c.request("/x", ident), (e: unknown) => e instanceof ApiError && e.code === "http_error" && !e.message.includes("<html>"));
  const ok = createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(200, null, [], "not json") });
  await assert.rejects(ok.request("/x", ident), (e: unknown) => e instanceof ApiError && e.code === "bad_response");
});

test("allowStatus lets readiness 503 through as data", async () => {
  const body = { ready: false, checks: { database: "ok", model: "unavailable" } };
  const api = makeApi(createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(503, body) }));
  const r = await api.ready();
  assert.equal(r.ready, false); assert.equal(r.checks.model, "unavailable");
});

test("network failure becomes NetworkError", async () => {
  const f = (async () => { throw new TypeError("fetch failed"); }) as unknown as typeof fetch;
  const c = createClient({ baseUrl: "/api/v1", fetchImpl: f });
  await assert.rejects(c.request("/health", ident), (e: unknown) => e instanceof NetworkError && /unreachable/i.test(e.message));
});

test("timeout aborts and reports a timeout, not a crash", async () => {
  const f = ((_u: string, init: RequestInit) => new Promise((_res, rej) => {
    init.signal!.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
  })) as unknown as typeof fetch;
  const c = createClient({ baseUrl: "/api/v1", fetchImpl: f, timeoutMs: 20 });
  await assert.rejects(c.request("/health", ident), (e: unknown) => e instanceof NetworkError && /in time/.test(e.message));
});

test("caller abort is propagated as the original abort error (not a NetworkError)", async () => {
  const f = ((_u: string, init: RequestInit) => new Promise((_res, rej) => {
    init.signal!.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
  })) as unknown as typeof fetch;
  const c = createClient({ baseUrl: "/api/v1", fetchImpl: f });
  const ac = new AbortController();
  const p = c.request("/health", ident, { signal: ac.signal });
  ac.abort();
  await assert.rejects(p, (e: unknown) => !(e instanceof NetworkError) && (e as Error).name === "AbortError");
});

test("endpoint wrappers build correct URLs, bodies and content types", async () => {
  const calls: Call[] = [];
  const pred = { transaction_id: 1, score: 0.5, threshold: 0.7, flagged: false, score_note: "n", warnings: [] };
  const api = makeApi(createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(200, pred, calls) }));
  await api.predict({ step: 1, type: "TRANSFER", amount: 5, nameOrig: "C1", nameDest: "C2" });
  assert.equal(calls[0]!.init.method, "POST");
  assert.equal((calls[0]!.init.headers as Record<string, string>)["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(calls[0]!.init.body as string), { step: 1, type: "TRANSFER", amount: 5, nameOrig: "C1", nameDest: "C2" });
  const page = { items: [], total: 0, page: 2, page_size: 10 };
  const api2 = makeApi(createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(200, page, calls) }));
  await api2.transactions({ page: 2, pageSize: 10, flagged: true, type: "", minScore: 0.5, sort: "score", order: "desc" });
  const u = new URL(calls[1]!.url, "http://x");
  assert.equal(u.pathname, "/api/v1/transactions");
  assert.deepEqual(Object.fromEntries(u.searchParams), { page: "2", page_size: "10", flagged: "true", min_score: "0.5", sort: "score", order: "desc" });
  const batch = { batch_id: "b", rows: 1, flagged: 0, results: [{ row: 2, transaction_id: 1, score: 0.1, flagged: false }] };
  const api3 = makeApi(createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(200, batch, calls) }));
  await api3.predictBatch("step,type\n1,A\n");
  assert.equal((calls[2]!.init.headers as Record<string, string>)["Content-Type"], "text/csv");
  assert.equal(calls[2]!.init.body, "step,type\n1,A\n");
});

test("a malformed success response is rejected by the guards, not rendered", async () => {
  const api = makeApi(createClient({ baseUrl: "/api/v1", fetchImpl: fakeFetch(200, { score: "high" }) }));
  await assert.rejects(api.predict({ step: 1, type: "A", amount: 1, nameOrig: "C1", nameDest: "C2" }), /Unexpected response/);
});
