"""Cost-based decision threshold on calibrated probabilities.

Decision rule: approve if p(default) < t.
  missed default (approved, TARGET=1)   costs LGD       * AMT_CREDIT
  rejected good client (rejected, TARGET=0) costs fp_margin * AMT_CREDIT
The threshold is chosen on the calibration split and the loss is then measured on the test split.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit.config import write_json  # noqa: E402
from credit.pipeline import ID, TARGET  # noqa: E402
from credit.train import mlflow_setup  # noqa: E402


def total_cost(y, p, credit, t: float, lgd: float, fp_margin: float) -> float:
    """Realised cost of the rule 'approve if p < t' given the true labels."""
    y, p, credit = np.asarray(y), np.asarray(p), np.asarray(credit, dtype=float)
    approve = p < t
    fn = approve & (y == 1)
    fp = ~approve & (y == 0)
    return float(lgd * credit[fn].sum() + fp_margin * credit[fp].sum())


def sweep(y, p, credit, lgd, fp_margin, grid) -> pd.DataFrame:
    return pd.DataFrame({"threshold": grid, "cost": [total_cost(y, p, credit, t, lgd, fp_margin) for t in grid]})


def best_threshold(y, p, credit, lgd, fp_margin, grid) -> float:
    s = sweep(y, p, credit, lgd, fp_margin, grid)
    return float(s.loc[s["cost"].idxmin(), "threshold"])


def _evaluate(test, t, lgd, m) -> dict:
    y, p, a = test[TARGET], test["prob_cal"], test["AMT_CREDIT"]
    approve = p < t
    return {
        "threshold": round(float(t), 4),
        "cost": total_cost(y, p, a, t, lgd, m),
        "approval_rate": float(approve.mean()),
        "default_rate_in_approved": float(y[approve].mean()) if approve.any() else None,
    }


def run_threshold(cfg: dict, run: str = "lgbm_all") -> dict:
    paths, c = cfg["paths"], cfg["cost"]
    preds = pd.read_parquet(Path(paths["preds_dir"]) / f"{run}_calibrated.parquet")
    amt = pd.read_parquet(paths["features"], columns=[ID, "AMT_CREDIT"])
    preds = preds.merge(amt, on=ID, how="left", validate="one_to_one")
    cal, test = preds[preds["split"] == "calib"], preds[preds["split"] == "test"]
    grid = np.round(np.arange(c["threshold_min"], c["threshold_max"] + 1e-9, c["threshold_step"]), 4)

    def scenario(lgd: float) -> dict:
        m = c["fp_margin"]
        t_star = best_threshold(cal[TARGET], cal["prob_cal"], cal["AMT_CREDIT"], lgd, m, grid)
        opt, half, allow = _evaluate(test, t_star, lgd, m), _evaluate(test, 0.5, lgd, m), _evaluate(test, 1.01, lgd, m)
        return {
            "lgd": lgd, "fp_margin": m,
            "bayes_threshold": round(m / (lgd + m), 4),  # optimum if probabilities are perfectly calibrated
            "optimal": opt, "at_0_5": half, "approve_all": allow,
            "reduction_vs_0_5_pct": 100 * (1 - opt["cost"] / half["cost"]) if half["cost"] else None,
            "reduction_vs_approve_all_pct": 100 * (1 - opt["cost"] / allow["cost"]) if allow["cost"] else None,
        }

    main = scenario(c["lgd"])
    sens = [scenario(lgd) for lgd in c["lgd_sensitivity"]]
    res = {"run": run, "selection_split": "calib", "evaluation_split": "test", "n_calib": len(cal),
           "n_test": len(test), "test_total_credit": float(test["AMT_CREDIT"].sum()), "main": main,
           "sensitivity": sens}

    results_dir = Path(paths["results_dir"])
    write_json(results_dir / "threshold.json", res)
    s_cal = sweep(cal[TARGET], cal["prob_cal"], cal["AMT_CREDIT"], c["lgd"], c["fp_margin"], grid)
    s_test = sweep(test[TARGET], test["prob_cal"], test["AMT_CREDIT"], c["lgd"], c["fp_margin"], grid)
    s_cal.merge(s_test, on="threshold", suffixes=("_calib", "_test")).to_csv(results_dir / "threshold_sweep.csv", index=False)

    fig_path = Path(paths["reports_dir"]) / "threshold_cost.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(s_cal["threshold"], s_cal["cost"] / s_cal["cost"].max(), label="calibration split (selection)")
    ax.plot(s_test["threshold"], s_test["cost"] / s_test["cost"].max(), label="test split (evaluation)")
    ax.axvline(main["optimal"]["threshold"], c="k", ls="--", lw=1, label=f"chosen t = {main['optimal']['threshold']:.2f}")
    ax.axvline(0.5, c="grey", ls=":", lw=1, label="t = 0.5")
    ax.set(xlabel="threshold on calibrated P(default)", ylabel="total cost (scaled to max = 1)",
           title=f"Cost sweep, LGD {c['lgd']}, lost margin {c['fp_margin']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    mlflow = mlflow_setup(cfg)
    with mlflow.start_run(run_name=f"threshold_{run}"):
        mlflow.log_params({"lgd": c["lgd"], "fp_margin": c["fp_margin"], "base_run": run, "selection_split": "calib"})
        mlflow.log_metrics({
            "threshold": main["optimal"]["threshold"],
            "test_cost_optimal": main["optimal"]["cost"],
            "test_cost_at_0_5": main["at_0_5"]["cost"],
            "test_cost_approve_all": main["approve_all"]["cost"],
        })
        mlflow.log_artifact(str(fig_path))
    return res
