"""Write a SYNTHETIC PaySim-shaped CSV for rehearsing the pipeline/UI before the real dataset is available.

    python scripts/make_synthetic_demo.py                  -> ml/data/raw/SYNTHETIC_demo.csv

The data is random with a planted, easy fraud pattern (large TRANSFER/CASH_OUT amounts). ANY metric computed
from it is meaningless. Always train with  --data-label "SYNTHETIC demo data"  so the UI shows its warning banners.
Delete the file before running the real EDA: ml/eda/inspect_dataset.py picks the FIRST csv in ml/data/raw/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from synthetic import make_synthetic  # noqa: E402

out = ROOT / "ml" / "data" / "raw" / "SYNTHETIC_demo.csv"
out.parent.mkdir(parents=True, exist_ok=True)
make_synthetic(n_steps=120, per_step=60, seed=7).to_csv(out, index=False)
print(f"wrote {out.name}: SYNTHETIC data, not PaySim. Do not report results from it.")
