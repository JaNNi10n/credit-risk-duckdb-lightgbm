"""SHAP TreeExplainer on test rows: global beeswarm + local waterfalls for approved/rejected applicants."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from credit.config import read_json, write_json  # noqa: E402
from credit.pipeline import ID  # noqa: E402
from credit.train import load_frame, mlflow_setup  # noqa: E402


def explain(cfg: dict, run: str = "lgbm_all") -> dict:
    paths, ex, seed = cfg["paths"], cfg["explain"], cfg["seed"]
    booster = lgb.Booster(model_file=str(Path(paths["models_dir"]) / f"{run}.txt"))
    cols = booster.feature_name()
    df = load_frame(cfg, cols)
    preds = pd.read_parquet(Path(paths["preds_dir"]) / f"{run}_calibrated.parquet", columns=[ID, "prob_cal"])
    test = df[df["split"] == "test"].merge(preds, on=ID, how="left")
    sample = test.sample(n=min(ex["n_rows"], len(test)), random_state=seed).reset_index(drop=True)
    X = sample[cols]

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)
    values = values[1] if isinstance(values, list) else values  # older shap returns [neg, pos]
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)

    # Categorical columns: numeric codes drive the colours, the original strings are displayed.
    numeric = X.apply(lambda s: s.cat.codes.where(s.notna()).astype(float) if isinstance(s.dtype, pd.CategoricalDtype) else s.astype(float))
    expl = shap.Explanation(values=values, base_values=np.full(len(X), base), data=numeric.to_numpy(),
                            display_data=X.astype(object).to_numpy(), feature_names=cols)

    reports = Path(paths["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(expl, max_display=ex["top_n"], show=False)
    plt.title(f"SHAP values (log-odds of default), {len(X):,} test applicants")
    plt.gcf().set_size_inches(9, 8)
    plt.tight_layout()
    plt.savefig(reports / "shap_beeswarm.png", dpi=110)
    plt.close("all")

    mean_abs = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(values).mean(axis=0)})
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    mean_abs.head(ex["top_n"]).to_csv(Path(paths["results_dir"]) / "shap_top20.csv", index=False)

    thr_file = Path(paths["results_dir"]) / "threshold.json"
    t = read_json(thr_file)["main"]["optimal"]["threshold"] if thr_file.exists() else 0.5
    approved = sample.index[sample["prob_cal"] < t].to_numpy()
    rejected = sample.index[sample["prob_cal"] >= t].to_numpy()
    rng = np.random.default_rng(seed)
    picks = {"approved": rng.choice(approved, size=min(2, len(approved)), replace=False),
             "rejected": rng.choice(rejected, size=min(2, len(rejected)), replace=False)}
    local = []
    for decision, rows in picks.items():
        for k, i in enumerate(sorted(rows), start=1):
            shap.plots.waterfall(expl[int(i)], max_display=12, show=False)
            sid, p = int(sample.loc[i, ID]), float(sample.loc[i, "prob_cal"])
            plt.title(f"{decision.upper()} SK_ID_CURR {sid}: calibrated P(default) {p:.3f}, threshold {t:.2f}", fontsize=10)
            plt.gcf().set_size_inches(9, 6)
            plt.tight_layout()
            fname = f"shap_waterfall_{decision}_{k}.png"
            plt.savefig(reports / fname, dpi=110)
            plt.close("all")
            local.append({"file": fname, "decision": decision, "SK_ID_CURR": sid, "prob_cal": round(p, 4)})

    res = {"run": run, "n_rows": len(X), "threshold": t, "base_value_logodds": base,
           "top_features": mean_abs.head(ex["top_n"]).to_dict("records"), "local_examples": local}
    write_json(Path(paths["results_dir"]) / "explain.json", res)

    mlflow = mlflow_setup(cfg)
    with mlflow.start_run(run_name=f"shap_{run}"):
        mlflow.log_params({"base_run": run, "n_rows": len(X), "explainer": "shap.TreeExplainer"})
        for f in ["shap_beeswarm.png", *[x["file"] for x in local]]:
            mlflow.log_artifact(str(reports / f))
    return res
