import { useState, type FormEvent } from "react";
import { useAuth } from "../api-context.tsx";
import { ApiError, NetworkError } from "../lib/api.ts";

export function Login() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      await login(username.trim(), password);
    } catch (e) {
      setError(e instanceof ApiError
        ? e.status === 401 ? "Incorrect username or password." : e.message
        : e instanceof NetworkError ? e.message : "Login failed. Please try again.");
    } finally { setBusy(false); }
  }

  return (
    <main className="login-shell">
      <section className="login-card">
        <div className="brand"><span className="logo" /> Fraud Operations</div>
        <p className="muted small">AIML-01 · Financial Fraud Detection Platform</p>
        <h1>Sign in</h1>
        <p className="muted">Access the fraud analyst workspace.</p>
        <form onSubmit={submit} className="login-form">
          <div>
            <label htmlFor="login-username">Username</label>
            <input id="login-username" autoComplete="username" value={username}
              onChange={e => setUsername(e.target.value)} required minLength={3} maxLength={80} />
          </div>
          <div>
            <label htmlFor="login-password">Password</label>
            <input id="login-password" type="password" autoComplete="current-password"
              value={password} onChange={e => setPassword(e.target.value)}
              required minLength={8} maxLength={128} />
          </div>
          {error && <div className="banner bad" role="alert">{error}</div>}
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="hint">Use the account provisioned by your platform administrator.</p>
      </section>
    </main>
  );
}
