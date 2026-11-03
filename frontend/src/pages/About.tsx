import { PageHead } from "../components/PageHead.tsx";
import { Card } from "../components/ui.tsx";

const NODES: { t: string; d: string; real: boolean }[] = [
  { t: "CSV / dataset file", d: "validated local file", real: true }, { t: "Replay (python -m streaming)", d: "API mode: no Kafka needed", real: true },
  { t: "Kafka (optional)", d: "client written, broker not verified", real: false }, { t: "FastAPI service", d: "validation · scoring · SQLite", real: true },
  { t: "Model artifact", d: "XGBoost / sklearn, hash-checked", real: true }, { t: "Monitoring (PSI)", d: "reference saved with the model", real: true },
  { t: "Feast (optional)", d: "config + tests, store not applied", real: false }, { t: "Graph (optional)", d: "analysis only, no GNN", real: false },
];

export function About() {
  return (
    <>
      <PageHead title="About, architecture & limitations" sub="AIML-01: real-time financial fraud detection platform (capstone project)." />
      <div className="grid">
        <Card title="Architecture">
          <div className="diagram">{NODES.map((n) => <div key={n.t} className={`node ${n.real ? "real" : ""}`}><b>{n.t}</b>{n.d}</div>)}</div>
          <p className="small muted">Solid outline = implemented and exercised in tests; dashed = optional, written but not verified against real infrastructure. Dataflow: file → (Kafka or direct) → API → model → SQLite → this dashboard.</p>
        </Card>
        <Card title="How to read the numbers">
          <ul>
            <li><b>Scores are ranking scores</b> from a class-weighted model. They are not calibrated probabilities. “Flagged” means the score reached a threshold chosen on validation data.</li>
            <li>The dashboard shows only what the API stored. Empty means empty; nothing is simulated.</li>
            <li>Model metrics come from the training run's held-out split and carry an origin label. Metrics from synthetic data are marked as such and prove nothing about real fraud.</li>
            <li>Drift (PSI) compares recent scored transactions with a reference saved at training time. Thresholds are rules of thumb, not statistical tests.</li>
          </ul>
        </Card>
        <Card title="Known limitations">
          <ul>
            <li><b>No authentication or authorization.</b> Run the API on localhost only; case and retraining controls are unprotected.</li>
            <li>Real-data training with PaySim has not been completed in this project state; any model shown may come from test fixtures.</li>
            <li>PaySim is simulated data and has only account names and a time step: no device, card or IP identifiers. No account–device–card graph is built and no fraud rings are claimed.</li>
            <li>History features (previous activity of a sender) use only transactions earlier scored by this platform, so serving differs from training until a feature store is wired in.</li>
            <li>Promotion in the retraining workflow records a decision; it does not hot-swap the running model.</li>
            <li>Not production-hardened: no threat model, in-memory rate limiting, SQLite storage.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}
