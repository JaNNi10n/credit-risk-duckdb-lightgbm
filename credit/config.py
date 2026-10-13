"""Config loading. Paths in config.yaml are resolved against the folder that holds the file."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
SQL_DIR = REPO_ROOT / "features"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Read config.yaml, apply shallow per-section overrides, and turn paths into absolute Paths."""
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = copy.deepcopy(value)
    base = path.resolve().parent
    cfg["paths"] = {k: (Path(v) if Path(v).is_absolute() else base / v) for k, v in cfg["paths"].items()}
    tmp = cfg.get("duckdb", {}).get("temp_directory")
    if tmp:
        cfg["duckdb"]["temp_directory"] = Path(tmp) if Path(tmp).is_absolute() else base / tmp
    return cfg


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
