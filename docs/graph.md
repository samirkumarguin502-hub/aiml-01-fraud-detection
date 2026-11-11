# Graph analysis (optional)

Status: construction, structural analysis, a leak-free baseline and an optional CPU-only GraphSAGE training path exist.
PyTorch and PyTorch Geometric are optional dependencies and are not required by the core platform. The optional path has
not been run on real PaySim in this workspace.

## What the data supports
PaySim has only sender `nameOrig`, receiver `nameDest` (merchant ids start with `M`) and `step`. So the only relationship is
account → account money transfer. **There are no device, card or IP identifiers**, so no account–device–card graph is
built and no fraud rings are asserted. If another dataset has such columns they are reported as present-but-unused.
`nameOrig` rarely repeats in PaySim, so the graph is expected to be sparse; check the real summary before drawing conclusions.

## What exists
- `python -m ml.graph.analyze --data <csv> [--max-step N]` prints, from the real file: nodes, distinct edges, degrees,
  weak components, repeat-sender fraction.
- Causal receiver features `dest_prev_txn_count` and `dest_prev_unique_senders` (strictly earlier steps only).
- `evaluate_graph_baseline`: ranking quality (PR-AUC / ROC-AUC vs prevalence) of those raw features on the same time-split
  **test** range the tabular model uses, with nothing fitted on test.
- `python -m ml.graph.analyze --data <csv> --train-gnn --gnn-epochs 100 [--gnn-out <model.pt>]` trains optional CPU GraphSAGE.
  Training edges are restricted to the chronological training range; validation/test node scores do not add graph edges.

## Not done / how to continue
A GNN would need an edge-classification setup (labels are per transaction), a time-respecting neighbour sampler, and a
comparison against XGBoost on the same split. Any blend with the XGBoost score must be tuned on the validation range and
reported once on test; no improvement is claimed.
