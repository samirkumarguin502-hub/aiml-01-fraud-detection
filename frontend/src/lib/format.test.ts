import assert from "node:assert/strict";
import { test } from "node:test";
import { fmtDate, fmtInt, fmtMetric, fmtRate, fmtScore, label, shortHash } from "./format.ts";

test("scores are plain numbers, never percentages", () => {
  assert.equal(fmtScore(0.81234), "0.8123");
  assert.ok(!fmtScore(0.9).includes("%"));
  assert.equal(fmtScore(null), "n/a"); assert.equal(fmtScore(NaN), "n/a"); assert.equal(fmtMetric(undefined), "n/a");
});
test("rates are percentages and null-safe", () => { assert.equal(fmtRate(0.1234), "12.34%"); assert.equal(fmtRate(null), "n/a"); });
test("misc formatting", () => {
  assert.equal(fmtInt(1234567), "1,234,567");
  assert.equal(fmtDate("2026-10-28T09:46:00+00:00"), "2026-10-28 09:46:00 UTC");
  assert.equal(fmtDate("garbage"), "n/a"); assert.equal(fmtDate(null), "n/a");
  assert.equal(label("false_positive"), "false positive"); assert.equal(shortHash("abcdef0123456789"), "abcdef0123");
});
