import os
from pathlib import Path

import pytest
import yaml

from credit.config import DEFAULT_CONFIG, load_config
from credit.pipeline import build_features, make_splits

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
FIXTURES = Path(__file__).parent / "fixtures"


def make_cfg(tmp: Path, raw_dir: Path, **sections) -> tuple[Path, dict]:
    """Write a config.yaml into tmp whose outputs all live under tmp; returns (path, loaded cfg)."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["paths"] = {
        "raw_dir": str(raw_dir),
        "features": str(tmp / "data" / "features.parquet"),
        "feature_groups": str(tmp / "data" / "feature_groups.json"),
        "splits": str(tmp / "data" / "splits.parquet"),
        "preds_dir": str(tmp / "data" / "preds"),
        "models_dir": str(tmp / "models"),
        "results_dir": str(tmp / "results"),
        "reports_dir": str(tmp / "reports"),
        "mlruns": str(tmp / "mlruns"),
    }
    raw["duckdb"]["threads"] = 2
    raw["mlflow"]["log_models"] = False
    for key, value in sections.items():
        raw[key] = {**raw[key], **value} if isinstance(raw.get(key), dict) else value
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path, load_config(path)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Feature store + splits built once from the committed fixture CSVs."""
    tmp = tmp_path_factory.mktemp("built")
    _, cfg = make_cfg(tmp, FIXTURES)
    summary = build_features(cfg)
    make_splits(cfg)
    return cfg, summary
