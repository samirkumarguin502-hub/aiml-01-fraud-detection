// Typed fetch client: timeouts, typed errors, no credentials, no logging of request bodies.
export class ApiError extends Error {
  readonly status: number; readonly code: string; readonly requestId: string | undefined; readonly details: unknown;
  constructor(status: number, code: string, message: string, requestId?: string, details?: unknown) {
    super(message); this.name = "ApiError"; this.status = status; this.code = code; this.requestId = requestId; this.details = details;
  }
}
export class NetworkError extends Error {
  constructor(message = "The API is unreachable. Is the backend running?") { super(message); this.name = "NetworkError"; }
}

export interface ClientOptions { baseUrl: string; fetchImpl?: typeof fetch; timeoutMs?: number; getToken?: () => string | null }
export interface RequestOptions { method?: "GET" | "POST" | "PUT" | "PATCH"; json?: unknown; body?: string; contentType?: string; allowStatus?: number[]; signal?: AbortSignal }

function errorFrom(status: number, data: unknown): ApiError {
  const e = (typeof data === "object" && data !== null ? (data as { error?: unknown }).error : undefined) as
    | { code?: unknown; message?: unknown; request_id?: unknown; details?: unknown } | undefined;
  const code = typeof e?.code === "string" ? e.code : "http_error";
  const message = typeof e?.message === "string" ? e.message : `Request failed (HTTP ${status}).`;
  return new ApiError(status, code, message, typeof e?.request_id === "string" ? e.request_id : undefined, e?.details);
}

export function createClient({ baseUrl, fetchImpl, timeoutMs = 15000, getToken }: ClientOptions) {
  const base = baseUrl.replace(/\/+$/, "");
  async function request<T>(path: string, parse: (v: unknown) => T, opts: RequestOptions = {}): Promise<T> {
    const doFetch = fetchImpl ?? fetch;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    let callerAborted = false;
    opts.signal?.addEventListener("abort", () => { callerAborted = true; ctrl.abort(); }, { once: true });
    const headers: Record<string, string> = { Accept: "application/json" };
    const token = getToken?.();
    if (token) headers.Authorization = `Bearer ${token}`;
    let body: string | undefined;
    if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
    else if (opts.body !== undefined) { headers["Content-Type"] = opts.contentType ?? "text/plain"; body = opts.body; }
    let res: Response;
    try {
      res = await doFetch(base + path, { method: opts.method ?? "GET", headers, ...(body !== undefined ? { body } : {}), signal: ctrl.signal, credentials: "omit" });
    } catch (e) {
      if (callerAborted) throw e;
      throw ctrl.signal.aborted ? new NetworkError("The API did not respond in time.") : new NetworkError();
    } finally { clearTimeout(timer); }
    let data: unknown = null;
    const text = await res.text();
    if (text) { try { data = JSON.parse(text); } catch { data = null; } }
    if (!res.ok && !(opts.allowStatus ?? []).includes(res.status)) throw errorFrom(res.status, data);
    if (data === null) throw new ApiError(res.status, "bad_response", "The API returned a non-JSON response.");
    return parse(data);
  }
  return { request };
}
export type Client = ReturnType<typeof createClient>;
