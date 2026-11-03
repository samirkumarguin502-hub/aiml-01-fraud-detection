import assert from "node:assert/strict";
import { test } from "node:test";
import { ApiError, NetworkError } from "./api.ts";
import { describeError } from "./errors.ts";
import { ResponseShapeError } from "./guards.ts";

test("errors map to safe views", () => {
  assert.equal(describeError(new NetworkError()).kind, "offline");
  const a = describeError(new ApiError(503, "model_unavailable", "No model", "r1", { model_state: "unavailable" }));
  assert.equal(a.title, "Service unavailable"); assert.equal(a.code, "model_unavailable"); assert.equal(a.requestId, "r1");
  assert.equal(describeError(new ApiError(429, "rate_limited", "slow")).title, "Too many requests");
  assert.equal(describeError(new ApiError(422, "validation_error", "bad")).title, "Request rejected");
  assert.equal(describeError(new ResponseShapeError("x")).kind, "shape");
  const u = describeError(new Error("secret stack /home/x"));
  assert.equal(u.kind, "unknown"); assert.ok(!u.message.includes("secret"));
});
