import { ApiError, NetworkError } from "./api.ts";
import { ResponseShapeError } from "./guards.ts";

export interface ErrorView { kind: "offline" | "api" | "shape" | "unknown"; title: string; message: string; code?: string; requestId?: string; details?: unknown }

// Turns any thrown value into something safe and useful to show. Never exposes stacks.
export function describeError(e: unknown): ErrorView {
  if (e instanceof NetworkError) return { kind: "offline", title: "Cannot reach the API", message: e.message };
  if (e instanceof ApiError) {
    const title = e.status === 503 ? "Service unavailable" : e.status === 429 ? "Too many requests" : e.status === 422 || e.status === 400 || e.status === 413 || e.status === 415 ? "Request rejected" : `API error (HTTP ${e.status})`;
    return { kind: "api", title, message: e.message, code: e.code, ...(e.requestId ? { requestId: e.requestId } : {}), ...(e.details !== undefined ? { details: e.details } : {}) };
  }
  if (e instanceof ResponseShapeError) return { kind: "shape", title: "Unexpected API response", message: e.message };
  return { kind: "unknown", title: "Something went wrong", message: "An unexpected error occurred." };
}
