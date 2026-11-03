// Client-side validation mirrors backend/app/validation.py. The server remains authoritative.
export const MAX_STEP = 1_000_000;
export const MAX_AMOUNT = 1e12;
export const MAX_UPLOAD_BYTES = 5_000_000;
const TYPE_RE = /^[A-Za-z0-9_-]{1,32}$/;
const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;
const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

export const BASE_FIELDS = ["step", "type", "amount", "nameOrig", "nameDest"] as const;
export const BALANCE_FIELDS = ["oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest"] as const;
export type BalanceField = (typeof BALANCE_FIELDS)[number];

export interface TxInput { step: number; type: string; amount: number; nameOrig: string; nameDest: string; oldbalanceOrg?: number; oldbalanceDest?: number; newbalanceOrig?: number; newbalanceDest?: number }
export type TxFields = Record<(typeof BASE_FIELDS)[number], string> & Partial<Record<BalanceField, string>>;
export type TxResult = { ok: true; value: TxInput } | { ok: false; errors: Partial<Record<keyof TxFields, string>> };

// Inputs the loaded model needs, taken from GET /model. Anything odd falls back to the base five fields
// (the server stays authoritative and will reject a mismatch with a clear message).
export function inputFieldsOf(info: { state: string; model: null | { input_fields?: string[] | undefined } } | undefined): readonly string[] {
  const f = info?.state === "ready" ? info.model?.input_fields : undefined;
  const known = new Set<string>([...BASE_FIELDS, ...BALANCE_FIELDS]);
  if (!Array.isArray(f) || !BASE_FIELDS.every((b) => f.includes(b)) || !f.every((x) => known.has(x))) return BASE_FIELDS;
  return [...BASE_FIELDS, ...BALANCE_FIELDS.filter((b) => f.includes(b))];
}

export function validateTransactionForm(f: TxFields, fields: readonly string[] = BASE_FIELDS): TxResult {
  const errors: Partial<Record<keyof TxFields, string>> = {};
  const step = f.step.trim(), amount = f.amount.trim();
  if (step === "") errors.step = "Required.";
  else if (!/^\d+$/.test(step) || Number(step) > MAX_STEP) errors.step = `Whole number between 0 and ${MAX_STEP}.`;
  if (amount === "") errors.amount = "Required.";
  else if (!NUM_RE.test(amount) || !Number.isFinite(Number(amount)) || Number(amount) < 0 || Number(amount) > MAX_AMOUNT) errors.amount = "Number between 0 and 1e12.";
  if (!TYPE_RE.test(f.type.trim())) errors.type = "1-32 letters, digits, _ or -.";
  if (!NAME_RE.test(f.nameOrig.trim())) errors.nameOrig = "1-64 letters, digits, _ or -.";
  if (!NAME_RE.test(f.nameDest.trim())) errors.nameDest = "1-64 letters, digits, _ or -.";
  const balances: Partial<Record<BalanceField, number>> = {};
  for (const b of BALANCE_FIELDS) {
    if (!fields.includes(b)) continue;
    const raw = (f[b] ?? "").trim();
    if (raw === "") errors[b] = "Required by the loaded model.";
    else if (!NUM_RE.test(raw) || !Number.isFinite(Number(raw)) || Number(raw) < 0 || Number(raw) > MAX_AMOUNT) errors[b] = "Number between 0 and 1e12.";
    else balances[b] = Number(raw);
  }
  if (Object.keys(errors).length) return { ok: false, errors };
  return { ok: true, value: { step: Number(step), type: f.type.trim(), amount: Number(amount), nameOrig: f.nameOrig.trim(), nameDest: f.nameDest.trim(), ...balances } };
}

export function validateCsvFile(file: { name: string; size: number }, maxBytes = MAX_UPLOAD_BYTES): string | null {
  if (!/\.csv$/i.test(file.name)) return "Choose a .csv file.";
  if (file.size === 0) return "The file is empty.";
  if (file.size > maxBytes) return `The file is larger than the ${Math.round(maxBytes / 1_000_000)} MB limit.`;
  return null;
}

export function validateNote(note: string, max = 500): string | null {
  return note.length > max ? `Keep it under ${max} characters.` : null;
}
