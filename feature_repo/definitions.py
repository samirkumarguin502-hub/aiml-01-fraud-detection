"""Feast definitions (requires `pip install feast pyarrow`). NOT executed in the authoring sandbox.

Workflow (PowerShell, from the project root; see docs/feast.md):
  python -m ml.feast_export --data ml\\data\\raw\\<file>.csv --out feature_repo\\data\\orig_history.parquet
  cd feature_repo ; feast apply
Only history features keyed by the sender account are served. Stateless features are computed from the request.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

account = Entity(name="account", join_keys=["nameOrig"])

orig_history_source = FileSource(path="data/orig_history.parquet", timestamp_field="event_timestamp")

orig_history = FeatureView(
    name="orig_history",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="orig_prev_txn_count", dtype=Float64),
        Field(name="orig_prev_amount_sum", dtype=Float64),
    ],
    source=orig_history_source,
)
