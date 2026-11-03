import assert from "node:assert/strict";
import { test } from "node:test";
import { BASE_FIELDS, inputFieldsOf, validateCsvFile, validateNote, validateTransactionForm, type TxFields } from "./validate.ts";

const good: TxFields = { step: "5", type: "TRANSFER", amount: "1234.5", nameOrig: "C1", nameDest: "C2" };

test("valid form converts to typed numbers", () => {
  const r = validateTransactionForm({ ...good, step: " 5 ", amount: "1e3" });
  assert.ok(r.ok); if (r.ok) assert.deepEqual(r.value, { step: 5, type: "TRANSFER", amount: 1000, nameOrig: "C1", nameDest: "C2" });
});
test("invalid fields are reported per field", () => {
  const cases: [Partial<TxFields>, string][] = [
    [{ step: "" }, "step"], [{ step: "-1" }, "step"], [{ step: "1.5" }, "step"], [{ step: "2000000" }, "step"], [{ step: "abc" }, "step"],
    [{ amount: "" }, "amount"], [{ amount: "-5" }, "amount"], [{ amount: "NaN" }, "amount"], [{ amount: "Infinity" }, "amount"], [{ amount: "1e13" }, "amount"], [{ amount: "5 dollars" }, "amount"],
    [{ type: "" }, "type"], [{ type: "A".repeat(33) }, "type"], [{ nameOrig: "x'; DROP" }, "nameOrig"], [{ nameDest: "a b" }, "nameDest"], [{ nameOrig: "../etc" }, "nameOrig"],
  ];
  for (const [patch, field] of cases) {
    const r = validateTransactionForm({ ...good, ...patch });
    assert.ok(!r.ok, JSON.stringify(patch)); if (!r.ok) assert.ok(r.errors[field as keyof TxFields], `${field} for ${JSON.stringify(patch)}`);
  }
});
test("csv file checks", () => {
  assert.equal(validateCsvFile({ name: "a.csv", size: 10 }), null);
  assert.match(validateCsvFile({ name: "a.txt", size: 10 })!, /csv/);
  assert.match(validateCsvFile({ name: "a.csv", size: 0 })!, /empty/);
  assert.match(validateCsvFile({ name: "a.csv", size: 6_000_000 })!, /larger/);
  assert.match(validateCsvFile({ name: "A.CSV.exe", size: 5 })!, /csv/);
});
test("note length", () => { assert.equal(validateNote("x"), null); assert.ok(validateNote("x".repeat(501))); });

const PRE = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "oldbalanceDest"];
test("balance fields are required and validated only when the model needs them", () => {
  const withBal: TxFields = { ...good, oldbalanceOrg: "5000", oldbalanceDest: "0" };
  const r = validateTransactionForm(withBal, PRE);
  assert.ok(r.ok); if (r.ok) assert.deepEqual(r.value, { step: 5, type: "TRANSFER", amount: 1234.5, nameOrig: "C1", nameDest: "C2", oldbalanceOrg: 5000, oldbalanceDest: 0 });
  const missing = validateTransactionForm(good, PRE);
  assert.ok(!missing.ok); if (!missing.ok) assert.deepEqual(Object.keys(missing.errors).sort(), ["oldbalanceDest", "oldbalanceOrg"]);
  for (const bad of ["-1", "abc", "NaN", "1e13"]) assert.ok(!validateTransactionForm({ ...withBal, oldbalanceOrg: bad }, PRE).ok, bad);
  const base = validateTransactionForm(withBal);  // a base model never receives balance values, even if typed in
  assert.ok(base.ok); if (base.ok) assert.ok(!("oldbalanceOrg" in base.value));
});
test("inputFieldsOf trusts only a well-formed list from a ready model", () => {
  assert.deepEqual(inputFieldsOf({ state: "ready", model: { input_fields: PRE } }), PRE);
  assert.deepEqual(inputFieldsOf(undefined), BASE_FIELDS);
  assert.deepEqual(inputFieldsOf({ state: "unavailable", model: null }), BASE_FIELDS);
  assert.deepEqual(inputFieldsOf({ state: "ready", model: { input_fields: ["step", "evil"] } }), BASE_FIELDS);
  assert.deepEqual(inputFieldsOf({ state: "ready", model: { input_fields: ["oldbalanceOrg"] } }), BASE_FIELDS);
  assert.deepEqual(inputFieldsOf({ state: "ready", model: {} }), BASE_FIELDS);
});
