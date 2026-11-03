import assert from "node:assert/strict";
import { test } from "node:test";
import { batchResultsCsv, csvCell, toCsv } from "./csv.ts";

test("formula-like text is neutralised, numbers are not", () => {
  for (const s of ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx"]) assert.ok(csvCell(s).startsWith("'") || csvCell(s).startsWith(`"'`), s);
  assert.equal(csvCell(-5), "-5"); assert.equal(csvCell(0.25), "0.25");
});
test("quoting and escaping", () => {
  assert.equal(csvCell('a,"b"'), '"a,""b"""'); assert.equal(csvCell("line1\nline2"), '"line1\nline2"'); assert.equal(csvCell(null), "");
});
test("batch results csv has stable header and rows", () => {
  const out = batchResultsCsv([{ row: 2, transaction_id: 7, score: 0.5, flagged: true }, { row: 3, transaction_id: 8, score: 0.01, flagged: false }]);
  assert.equal(out, "csv_row,transaction_id,ranking_score,flagged\r\n2,7,0.5,yes\r\n3,8,0.01,no\r\n");
  assert.equal(toCsv([], ["a"]), "a\r\n");
});
