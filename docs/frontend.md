# Frontend (React + TypeScript + Vite)

Status: **never built or rendered.** `npm install` is impossible in the authoring sandbox (no network), so the UI has not
been opened in a browser. What *was* executed: all logic modules (`src/lib`) are tested with Node's built-in runner
(27 tests) and type-checked under `strict`; the components were type-checked against a typed React shim (weaker than real
`@types/react`). The component tests (`*.test.tsx`, Vitest + Testing Library) are written but **not run**.

## Run (Windows PowerShell)
```
cd frontend
npm install
npm run dev          # http://localhost:5173 ; /api is proxied to http://127.0.0.1:8000 (override: $env:VITE_API_PROXY)
npm run test:lib     # runs without extra packages on Node >= 22
npm test             # lib tests + component tests (needs npm install)
npm run build        # tsc + vite build
```
Start the backend first (docs/api.md). The dashboard still loads without it and shows an explicit "API offline" state.

## Pages and the endpoints they use
| Page | Endpoints |
|---|---|
| Overview | `GET /dashboard/summary` |
| Alerts & cases | `GET /alerts`, `GET/PATCH/PUT /transactions/{id}[/case|/feedback]` |
| Transactions | `GET /transactions` (filters, sort, pagination), detail as above |
| Live / replay | `GET /transactions`, `GET /pipeline/status` (polls every 5 s) |
| Score one | `POST /predict` |
| Batch CSV | `POST /predict/batch` (raw `text/csv`), results download built client-side |
| Model & thresholds | `GET /model` |
| Drift & monitoring | `GET /monitoring/status`, `GET/POST /monitoring/retraining[...]` |
| System health | `GET /health`, `GET /ready`, `GET /pipeline/status` |
| About & limits | static |

## Truthfulness rules implemented
Every number comes from an API response that passes runtime shape checks (`src/lib/guards.ts`); a response that fails
them is shown as an error. No fixtures are bundled. Empty database → "No transactions scored yet". No model → no metrics.
Synthetic-trained models trigger a visible warning. Scores are printed as plain numbers with a "not a calibrated
probability" note; percentages are used only for rates (flag rate, flagged share). No secrets exist in the bundle;
`credentials: "omit"` on every request. CSV downloads neutralise formula-like cells.

## Known gaps
Not browser-tested (layout, responsiveness, accessibility were designed in, not audited). No authentication UI because the API
has none. No i18n. Charts are plain SVG.

## Pass 6 changes
*Score one* and *Batch CSV* read `GET /model` (`input_fields`) and show/require balance fields only when the loaded model needs them; *Model & thresholds* additionally shows the feature set, training time/memory, the classification report, precision-recall and ROC curves and feature importance **only when the artifact contains them** (otherwise an explicit empty state). Component tests for these were updated but not run.
