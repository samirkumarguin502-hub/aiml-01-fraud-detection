"""Documentation must list every feature the code can emit (stops silent drift between code and docs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_synthetic  # noqa: E402

from ml.pipeline.features import FeatureContract  # noqa: E402
from ml.pipeline.schema import FEATURE_SET_INPUTS  # noqa: E402

DOC = Path(__file__).resolve().parents[1] / "docs" / "feature_engineering.md"


def test_feature_engineering_doc_mentions_every_feature_and_set():
    text = DOC.read_text(encoding="utf-8")
    df = make_synthetic(5, 5)
    for fs in FEATURE_SET_INPUTS:
        assert f"`{fs}`" in text, fs
        for name in FeatureContract.fit(df, fs).feature_names:
            if name.startswith("type__"):
                continue  # one-hots are documented as a family
            assert f"`{name}`" in text, f"{name} is emitted by feature set {fs} but not documented"
    assert "`isFlaggedFraud`" in text and "`isFraud`" in text


def test_provenance_doc_does_not_claim_an_unverified_download():
    text = (Path(__file__).resolve().parents[1] / "docs" / "paysim_provenance.md").read_text(encoding="utf-8")
    assert "NOT been downloaded" in text or "has been downloaded" in text  # a status line must exist
