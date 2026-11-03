# Security review (author's self-review, 30 Oct 2026)

**Scope and method.** Source review of `backend/`, `ml/`, `monitoring/`, `streaming/`, `frontend/src`, Docker/K8s/CI files; a pattern scan
(`eval/exec/os.system/subprocess/shell=True/pickle/yaml.load/input(`, hard-coded secrets, debug/reload flags, `dangerouslySetInnerHTML/innerHTML/
localStorage`); unit tests of the validation, limits and error handling listed below; a throwaway `git init` to confirm ignored files are not tracked.
**Not done:** penetration test, dependency vulnerability scan (no network), threat model, review by a second person, running the live HTTP layer.
This is not a certification that the application is secure.

## Headline
**There is no authentication or authorization.** Anyone who can reach the API port can score transactions, label data, change case
status and advance retraining requests. Compose binds to `127.0.0.1`; keep it that way. Do not expose this service to a network.

## Controls in place (with evidence)
| Area | Control | Evidence |
|---|---|---|
| Input validation | One validator for JSON and CSV: types, ranges, lengths, regexes; unknown JSON fields rejected; extra CSV columns never read | `tests/test_backend_core.py` (validation + CSV tests) |
| Injection | Parametrised SQL only; sort columns whitelisted; names restricted to `[A-Za-z0-9_-]` | sort-injection and name-injection tests |
| Uploads | Raw body streamed with a hard byte cap, row cap, UTF-8 check, header/duplicate/ragged-row checks; **never written to disk**, so no temp-file or path-traversal surface | CSV tests; code review of `main.py` (HTTP test not run) |
| Paths | No user-supplied path reaches the filesystem. Model/DB/artifact locations come from environment variables; retraining `candidate` must match `[A-Za-z0-9_-]{1,80}`; artifact metadata cannot redirect file loads | traversal tests in `test_training_pipeline.py`, `test_monitoring.py` |
| Error handling | Uniform `{"error":{code,message,request_id}}`; no stack traces, no echoed values, no filesystem paths in model errors | value-echo and path-leak tests |
| Logging | Log fields are whitelisted scalars; transaction fields cannot be logged | `test_logs_contain_no_transaction_data` |
| Model files | SHA-256 + library-version + feature-contract checks before loading; XGBoost uses native JSON, not pickle | tamper/corrupt/mismatch tests |
| Rate limiting | In-memory sliding window per client (predict, batch, read) with `Retry-After` | limiter unit tests; HTTP 429 test **not run** |
| CORS | Explicit origins only, `*` rejected at startup, no credentials | config tests; HTTP CORS test **not run** |
| Secrets | None required or present; `.env.example` placeholders only; frontend contains no keys and sends `credentials: "omit"` | pattern scan |
| Frontend | No `innerHTML`/`dangerouslySetInnerHTML`/`localStorage`; responses shape-checked before rendering; CSV export neutralises formula injection | scan + `csv.test.ts`, `guards.test.ts` |
| Containers | Non-root user, read-only models mount, loopback port binding, K8s `runAsNonRoot`, dropped capabilities, no privilege escalation | file review only; never built |
| Git hygiene | `.gitignore` covers venvs, `.env*`, datasets, artifacts, databases, `node_modules`, build output; verified with a throwaway repo | see testing report |

## Known risks and gaps (ordered by importance)
1. **No authentication** (above). Case/feedback/retraining endpoints are the most sensitive.
2. **Pickle-based model files** (`model.joblib`): loading executes code if the file is malicious. The SHA-256 detects accidental
   corruption and tampering with the file *relative to its own metadata*, but an attacker who can write both files defeats it.
   Only load artifacts you trained. Prefer `--model xgboost` (JSON format).
3. **In-memory rate limiter**: per process, resets on restart, trusts `X-Forwarded-For` only when `TRUST_PROXY=true`. Not a DDoS defence.
4. JSON requests **without** a `Content-Length` header are not size-checked by the middleware (the framework parses them).
5. SQLite stores account identifiers and amounts in plaintext, unencrypted, with no retention policy. PaySim identifiers are
   synthetic; real data would need privacy review.
6. Retraining "promotion" is a recorded decision, not enforced access control.
7. Dependencies are unpinned lower bounds (`>=`) and were not vulnerability-scanned. Run `pip-audit` and `npm audit` yourself.
8. Docker/K8s/CI files have not been executed, so their hardening settings are unproven.
9. Development proxy (Vite) is for local use only; the nginx config adds basic headers but no TLS or CSP.

## Before any real deployment
Add authentication + authorization, TLS, a real rate limiter/gateway, request-size limits at the proxy, pinned and scanned dependencies,
a threat model, data-retention rules, and a second-person review.

## Pass 6 additions
- Balance fields (when a balance model is loaded) are validated like amounts (finite, 0 to 1e12) and stored in SQLite in plaintext, like other transaction fields. Post-transaction balances are refused unless the loaded model was trained as the diagnostic set.
- Plots, `report.md` and `summary.json` contain aggregate statistics, the data label and a dataset hash only; the EDA stats file includes the first five dataset rows (simulated) and can be deleted before committing.
- No Kaggle credentials are requested, stored or referenced in code; the optional CLI download needs a token you keep in `%USERPROFILE%\.kaggle\`, outside the project (documented as unverified).
- `ml/artifacts/` stays git-ignored; sklearn artifacts remain pickle-based (see risk 2 above); an XGBoost artifact uses the native JSON format.
