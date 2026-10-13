"""Build reports/results.md (and the README results block) from the files in results/."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import subprocess
from pathlib import Path

import pandas as pd

from credit.config import REPO_ROOT, read_json, write_json

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LIBS = ["duckdb", "lightgbm", "scikit-learn", "shap", "mlflow", "pandas", "numpy", "pyarrow"]


def environment() -> dict:
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip() or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    ram = None
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return {"os": f"{platform.system()} {platform.release()} ({platform.version()})", "cpu": cpu,
            "logical_cores": os.cpu_count(), "ram_gb": ram, "python": platform.python_version(),
            "libraries": {lib: md.version(lib) for lib in LIBS}}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _f(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def _money(x):
    return f"{x / 1e6:,.1f} M"


def build_markdown(cfg: dict, img: str = "") -> str:
    """Markdown for every result file that exists. `img` prefixes image paths."""
    res = Path(cfg["paths"]["results_dir"])

    def load(name):
        p = res / name
        return read_json(p) if p.exists() else None

    out = []
    feats, splits = load("features_summary.json"), load("splits_summary.json")
    if feats and feats.get("synthetic"):
        out.append("> **SYNTHETIC DATA.** These numbers come from generated test data, not from Home Credit. "
                   "They only show that the code runs and say nothing about model quality.\n")
    if splits:
        rows = [{"split": s, "rows": f"{v['rows']:,}", "default rate": f"{v['default_rate']:.4f}"}
                for s, v in splits.items()]
        out += ["### Data split (stratified on TARGET, seed 42)", _md_table(pd.DataFrame(rows)), ""]
    if feats:
        rows = [{"table (SQL file)": t, "features": v["features"], "rows out": f"{v['rows']:,}", "seconds": v["seconds"]}
                for t, v in feats["tables"].items()]
        out += ["### Feature pipeline (DuckDB)", _md_table(pd.DataFrame(rows)), "",
                f"Total: **{feats['n_features_total']} features** for **{feats['rows']:,} applicants** "
                f"(one row per SK_ID_CURR), {feats['total_seconds']:.1f} s end to end "
                f"(join + Parquet write {feats['join_and_write_seconds']:.1f} s), DuckDB {feats['duckdb_version']}, "
                f"{feats['duckdb_threads']} threads.", ""]
    det = load("determinism.json")
    if det:
        out += [f"Determinism: {det['runs']} runs, identical SHA-256 = **{det['identical']}** "
                f"(`{det['hashes'][0][:16]}...`).", ""]

    runs_dir = res / "runs"
    runs = {p.stem: read_json(p) for p in sorted(runs_dir.glob("*.json"))} if runs_dir.exists() else {}
    main_runs = [r for r in ("logreg_app", "lgbm_app", "lgbm_all", "lgbm_all_nosensitive") if r in runs]
    if main_runs:
        rows = [{"run": r, "features": runs[r]["n_features"], "test ROC-AUC": _f(runs[r]["test_auc"]),
                 "test PR-AUC": _f(runs[r]["test_pr_auc"]), "test Brier": _f(runs[r]["test_brier"]),
                 "train s": runs[r]["train_seconds"], "trees": runs[r].get("best_iteration", "")} for r in main_runs]
        n_test = f", {splits['test']['rows']:,} rows" if splits else ""
        out += [f"### Models (test split{n_test})", _md_table(pd.DataFrame(rows)), ""]
        if "lgbm_all" in runs and "lgbm_all_nosensitive" in runs:
            d = runs["lgbm_all_nosensitive"]["test_auc"] - runs["lgbm_all"]["test_auc"]
            out += [f"Fairness check: dropping {', '.join(cfg['sensitive_features'])} changes test AUC by "
                    f"**{d:+.4f}** ({_f(runs['lgbm_all']['test_auc'])} -> {_f(runs['lgbm_all_nosensitive']['test_auc'])}).", ""]

    abl = load("ablation.json")
    if abl:
        df = pd.DataFrame(abl)[["step", "added", "n_features", "best_iteration", "test_auc", "delta_vs_previous",
                                "delta_vs_app_only", "train_seconds"]]
        out += ["### Table ablation (LightGBM, cumulative)", _md_table(df), ""]

    cal = load("calibration.json")
    if cal:
        rows = [{"": "raw LightGBM", "Brier": _f(cal["test_brier_before"], 5), "ECE (10 bins)": _f(cal["test_ece_before"], 5),
                 "ROC-AUC": _f(cal["test_auc_before"]), "mean predicted": _f(cal["test_mean_pred_before"])},
                {"": "isotonic (fit on calib split)", "Brier": _f(cal["test_brier_after"], 5),
                 "ECE (10 bins)": _f(cal["test_ece_after"], 5), "ROC-AUC": _f(cal["test_auc_after"]),
                 "mean predicted": _f(cal["test_mean_pred_after"])}]
        out += ["### Calibration (test split)", _md_table(pd.DataFrame(rows)), "",
                f"Observed default rate on test: {cal['test_default_rate']:.4f}. Brier change: {cal['brier_delta']:+.5f}.",
                "", f"![reliability diagram]({img}reliability.png)", ""]

    thr = load("threshold.json")
    if thr:
        m = thr["main"]
        rows = [{"policy": name, "threshold": _f(v["threshold"], 2) if v["threshold"] <= 1 else "none (approve all)",
                 "approval rate": f"{v['approval_rate']:.3f}",
                 "default rate in approved": _f(v["default_rate_in_approved"]),
                 "test cost": _money(v["cost"])}
                for name, v in (("cost-optimal", m["optimal"]), ("0.5 cut-off", m["at_0_5"]), ("approve all", m["approve_all"]))]
        out += [f"### Cost-based threshold (LGD {m['lgd']}, lost margin {m['fp_margin']})", _md_table(pd.DataFrame(rows)), "",
                f"Chosen on the calibration split, measured on the test split. The cost-optimal threshold cuts the "
                f"test cost by **{m['reduction_vs_0_5_pct']:.1f}%** vs the 0.5 cut-off and by "
                f"**{m['reduction_vs_approve_all_pct']:.1f}%** vs approving everyone. "
                f"Theory check: with perfectly calibrated probabilities the optimum is "
                f"margin / (LGD + margin) = {m['bayes_threshold']:.3f}.", ""]
        rows = [{"LGD": s["lgd"], "chosen threshold": _f(s["optimal"]["threshold"], 2),
                 "Bayes threshold": _f(s["bayes_threshold"], 3), "approval rate": f"{s['optimal']['approval_rate']:.3f}",
                 "cost optimal": _money(s["optimal"]["cost"]), "cost @0.5": _money(s["at_0_5"]["cost"]),
                 "cost approve-all": _money(s["approve_all"]["cost"]),
                 "saving vs 0.5": f"{s['reduction_vs_0_5_pct']:.1f}%",
                 "saving vs approve-all": f"{s['reduction_vs_approve_all_pct']:.1f}%"} for s in thr["sensitivity"]]
        out += ["LGD sensitivity (threshold re-optimised on the calibration split for each LGD):",
                _md_table(pd.DataFrame(rows)), "", f"![cost sweep]({img}threshold_cost.png)", ""]

    exp = load("explain.json")
    if exp:
        top = pd.DataFrame(exp["top_features"]).head(10)
        top["mean_abs_shap"] = top["mean_abs_shap"].map(lambda v: f"{v:.4f}")
        top.insert(0, "rank", range(1, len(top) + 1))
        out += [f"### SHAP (TreeExplainer, {exp['n_rows']:,} test rows)", _md_table(top), "",
                f"![SHAP beeswarm]({img}shap_beeswarm.png)", ""]
        out += [f"![{x['decision']} {x['SK_ID_CURR']}]({img}{x['file']})" for x in exp["local_examples"]]
        out.append("")

    timings = load("stage_timings.json")
    if timings:
        out += ["### Wall-clock time per stage (seconds)",
                _md_table(pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items()])), ""]
    env = load("environment.json")
    if env:
        out += [f"Hardware: {env['cpu']}, {env['logical_cores']} logical cores, {env['ram_gb']} GB RAM, "
                f"{env['os']}, Python {env['python']}. Libraries: "
                + ", ".join(f"{k} {v}" for k, v in env["libraries"].items()) + ".", ""]
    return "\n".join(out)


def write_report(cfg: dict, update_readme: bool = True) -> Path:
    res_dir = Path(cfg["paths"]["results_dir"])
    write_json(res_dir / "environment.json", environment())
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "results.md"
    path.write_text("# Results\n\nGenerated by `python -m credit report` from the files in `results/`.\n\n"
                    + build_markdown(cfg), encoding="utf-8")

    feats = res_dir / "features_summary.json"
    synthetic = feats.exists() and read_json(feats).get("synthetic")
    readme = REPO_ROOT / "README.md"
    if update_readme and not synthetic and readme.exists() and reports.resolve() == (REPO_ROOT / "reports").resolve():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            block = f"{START}\n{build_markdown(cfg, img='reports/')}\n{END}"
            readme.write_text(re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S),
                              encoding="utf-8")
    return path
