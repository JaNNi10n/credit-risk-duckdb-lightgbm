"""CLI: python -m credit {features,train,ablate,calibrate,threshold,explain,report,all}."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from credit.config import load_config, read_json, write_json  # noqa: E402
from credit.pipeline import GROUPS  # noqa: E402


def _timed(cfg: dict, stage: str, fn, *args, **kwargs):
    """Run one stage and record its wall-clock time in results/stage_timings.json."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    secs = round(time.perf_counter() - t0, 2)
    path = Path(cfg["paths"]["results_dir"]) / "stage_timings.json"
    timings = read_json(path) if path.exists() else {}
    timings[stage] = secs
    write_json(path, timings)
    print(f"[{stage}] done in {secs:.1f}s")
    return out


def _groups(arg: str) -> list[str]:
    groups = ["all"] if arg == "all" else [g.strip() for g in arg.split(",")]
    unknown = [g for g in groups if g != "all" and g not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown table group(s) {unknown}; choose from {list(GROUPS)} or 'all'")
    return groups


def cmd_features(cfg, a):
    from credit.pipeline import build_features, make_splits

    summary = _timed(cfg, "features", build_features, cfg)
    make_splits(cfg)
    if getattr(a, "check_determinism", False):
        from credit.pipeline import check_determinism

        det = _timed(cfg, "features_rebuild_for_determinism", check_determinism, cfg, summary["sha256"])
        print(f"determinism: identical parquet hash on rebuild = {det['identical']}")
    print(json.dumps({k: v for k, v in summary.items() if k != "tables"}, indent=2))
    for t, info in summary["tables"].items():
        print(f"  {t:<24} {info['features']:>4} features  {info['rows']:>8} rows  {info['seconds']:>6.1f}s")


def cmd_train(cfg, a):
    from credit.train import train_lgbm, train_logreg

    groups = _groups(a.tables)
    if a.model == "logreg":
        res = _timed(cfg, f"train_logreg_{a.tables}", train_logreg, cfg, groups)
    else:
        drop = cfg["sensitive_features"] if a.drop_sensitive else None
        suffix = "_nosensitive" if a.drop_sensitive else ""
        res = _timed(cfg, f"train_lgbm_{a.tables}{suffix}", train_lgbm, cfg, groups, drop, suffix)
    print({k: v for k, v in res.items() if k.endswith("auc") or k in ("name", "n_features", "train_seconds")})


def cmd_ablate(cfg, _):
    from credit.train import ablate

    for row in _timed(cfg, "ablate", ablate, cfg):
        print(row)


def cmd_calibrate(cfg, a):
    from credit.calibrate import calibrate

    print(_timed(cfg, "calibrate", calibrate, cfg, a.run))


def cmd_threshold(cfg, a):
    from credit.threshold import run_threshold

    res = _timed(cfg, "threshold", run_threshold, cfg, a.run)
    print(json.dumps(res["main"], indent=2))


def cmd_explain(cfg, a):
    from credit.explain import explain

    res = _timed(cfg, "explain", explain, cfg, a.run)
    print(f"top features: {[f['feature'] for f in res['top_features'][:10]]}")


def cmd_report(cfg, a):
    from credit.report import write_report

    print(f"wrote {write_report(cfg, update_readme=not a.no_readme)}")


def cmd_all(cfg, a):
    if not a.skip_features:
        cmd_features(cfg, a)
    cmd_train(cfg, argparse.Namespace(model="logreg", tables="app", drop_sensitive=False))
    cmd_ablate(cfg, a)  # its last step is LightGBM on all tables (run name lgbm_all)
    cmd_train(cfg, argparse.Namespace(model="lgbm", tables="all", drop_sensitive=True))
    cmd_calibrate(cfg, a)
    cmd_threshold(cfg, a)
    cmd_explain(cfg, a)
    cmd_report(cfg, a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m credit", description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: repo config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="DuckDB SQL -> data/features.parquet + data/splits.parquet")
    f.add_argument("--check-determinism", action="store_true", help="build twice and compare SHA-256")
    t = sub.add_parser("train", help="train one model")
    t.add_argument("--model", choices=["lgbm", "logreg"], default="lgbm")
    t.add_argument("--tables", default="all", help=f"'all' or comma list of {list(GROUPS)}")
    t.add_argument("--drop-sensitive", action="store_true", help="drop config.sensitive_features (fairness check)")
    sub.add_parser("ablate", help="cumulative per-table LightGBM ablation")
    for name in ("calibrate", "threshold", "explain"):
        p = sub.add_parser(name)
        p.add_argument("--run", default="lgbm_all")
    r = sub.add_parser("report", help="write reports/results.md (and the README results block)")
    r.add_argument("--no-readme", action="store_true")
    al = sub.add_parser("all", help="every stage in order")
    al.add_argument("--skip-features", action="store_true")
    al.add_argument("--check-determinism", action="store_true")
    al.add_argument("--run", default="lgbm_all")
    al.add_argument("--no-readme", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    {"features": cmd_features, "train": cmd_train, "ablate": cmd_ablate, "calibrate": cmd_calibrate,
     "threshold": cmd_threshold, "explain": cmd_explain, "report": cmd_report, "all": cmd_all}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
