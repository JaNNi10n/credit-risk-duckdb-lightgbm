"""Isotonic calibration fitted on the 10% calibration split only, evaluated on the test split."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from credit.config import write_json  # noqa: E402
from credit.pipeline import TARGET  # noqa: E402
from credit.train import mlflow_setup  # noqa: E402


def reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]: mean predicted vs observed default rate per bin."""
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index()
    out["bin_lo"] = out["bin"] / n_bins
    return out


def ece(table: pd.DataFrame) -> float:
    """Expected calibration error: bin-size weighted |mean predicted - observed|."""
    return float((table["n"] * (table["mean_pred"] - table["obs_rate"]).abs()).sum() / table["n"].sum())


def calibrate(cfg: dict, run: str = "lgbm_all") -> dict:
    paths, n_bins = cfg["paths"], cfg["calibration"]["n_bins"]
    preds = pd.read_parquet(Path(paths["preds_dir"]) / f"{run}.parquet")
    cal, test = preds[preds["split"] == "calib"], preds[preds["split"] == "test"]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cal["prob"], cal[TARGET])
    preds["prob_cal"] = iso.predict(preds["prob"])
    test_cal = preds.loc[preds["split"] == "test", "prob_cal"]

    rel_before = reliability(test[TARGET], test["prob"], n_bins)
    rel_after = reliability(test[TARGET], test_cal, n_bins)
    res = {
        "run": run,
        "n_calib": len(cal),
        "n_test": len(test),
        "test_brier_before": float(brier_score_loss(test[TARGET], test["prob"])),
        "test_brier_after": float(brier_score_loss(test[TARGET], test_cal)),
        "test_ece_before": ece(rel_before),
        "test_ece_after": ece(rel_after),
        "test_auc_before": float(roc_auc_score(test[TARGET], test["prob"])),
        "test_auc_after": float(roc_auc_score(test[TARGET], test_cal)),
        "test_mean_pred_before": float(test["prob"].mean()),
        "test_mean_pred_after": float(test_cal.mean()),
        "test_default_rate": float(test[TARGET].mean()),
    }
    res["brier_delta"] = res["test_brier_after"] - res["test_brier_before"]

    Path(paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, Path(paths["models_dir"]) / f"isotonic_{run}.joblib")
    preds.to_parquet(Path(paths["preds_dir"]) / f"{run}_calibrated.parquet", index=False)
    results_dir = Path(paths["results_dir"])
    pd.concat([rel_before.assign(stage="before"), rel_after.assign(stage="after")]).to_csv(
        results_dir / "reliability_bins.csv", index=False)
    write_json(results_dir / "calibration.json", res)

    fig_path = Path(paths["reports_dir"]) / "reliability.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    ax.plot(rel_before["mean_pred"], rel_before["obs_rate"], "o-", label=f"raw LightGBM (Brier {res['test_brier_before']:.4f})")
    ax.plot(rel_after["mean_pred"], rel_after["obs_rate"], "s-", label=f"isotonic (Brier {res['test_brier_after']:.4f})")
    ax.set(xlabel="mean predicted default probability (bin)", ylabel="observed default rate",
           title=f"Reliability on test split ({n_bins} equal-width bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    mlflow = mlflow_setup(cfg)
    with mlflow.start_run(run_name=f"calibration_{run}"):
        mlflow.log_params({"method": "isotonic", "fit_split": "calib", "base_run": run, "n_bins": n_bins})
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, float)})
        mlflow.log_artifact(str(fig_path))
    return res
