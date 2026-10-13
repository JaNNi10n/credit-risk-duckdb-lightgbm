"""Model training: Logistic Regression baseline and LightGBM, every run tracked in MLflow."""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit.config import write_json
from credit.pipeline import GROUPS, ID, TARGET, feature_columns

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
EXPERIMENT = "home-credit-default-risk"


# ----------------------------------------------------------------------------------- helpers
def run_name(model: str, groups: list[str], suffix: str = "") -> str:
    tag = "all" if groups == ["all"] or groups == list(GROUPS) else "+".join(groups)
    return f"{model}_{tag}{suffix}"


def categorical_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]


def load_frame(cfg: dict, cols: list[str]) -> pd.DataFrame:
    """Features + TARGET + split label; text columns become pandas categoricals."""
    paths = cfg["paths"]
    df = pd.read_parquet(paths["features"], columns=[ID, TARGET, *cols])
    df = df.merge(pd.read_parquet(paths["splits"]), on=ID, how="inner", validate="one_to_one")
    for c in categorical_columns(df, cols):
        df[c] = df[c].astype("category")
    return df


def score(y, p) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


def mlflow_setup(cfg: dict):
    """Local MLflow: sqlite backend + artifact folder, both under paths.mlruns.

    MLflow 3.x refuses the plain './mlruns' file store (maintenance mode), so the run metadata
    lives in mlruns/mlflow.db and the artefacts in mlruns/artifacts. Still fully local.
    """
    import mlflow

    root = Path(cfg["paths"]["mlruns"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (root / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _save_outputs(cfg, name, df, prob, metrics, cols):
    paths = cfg["paths"]
    out = df.loc[df["split"].isin(["calib", "test"]), [ID, TARGET, "split"]].copy()
    out["prob"] = prob[df["split"].isin(["calib", "test"]).to_numpy()]
    Path(paths["preds_dir"]).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(paths["preds_dir"]) / f"{name}.parquet", index=False)
    write_json(Path(paths["results_dir"]) / "runs" / f"{name}.json", {**metrics, "features": cols})


def _split_metrics(df, prob) -> dict:
    m = {}
    for split in ("calib", "test"):
        mask = (df["split"] == split).to_numpy()
        m.update({f"{split}_{k}": v for k, v in score(df.loc[mask, TARGET], prob[mask]).items()})
    return m


# ------------------------------------------------------------------------------------ models
def train_lgbm(cfg: dict, groups: list[str], drop: list[str] | None = None, suffix: str = "") -> dict:
    """LightGBM on the 70% train split; early stopping on an inner stratified 10% of it."""
    mlflow = mlflow_setup(cfg)
    hp, seed = cfg["lgbm"], cfg["seed"]
    name = run_name("lgbm", groups, suffix)
    cols = feature_columns(cfg, groups, drop)
    df = load_frame(cfg, cols)
    train = df[df["split"] == "train"]
    fit, val = train_test_split(train, test_size=hp["inner_valid_frac"], stratify=train[TARGET], random_state=seed)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": hp["learning_rate"],
        "num_leaves": hp["num_leaves"],
        "feature_fraction": hp["feature_fraction"],
        "min_child_samples": hp["min_child_samples"],
        "num_threads": hp["num_threads"],
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(fit[cols], fit[TARGET])
    dval = lgb.Dataset(val[cols], val[TARGET], reference=dtrain)
    t0 = time.perf_counter()
    booster = lgb.train(
        params, dtrain, num_boost_round=hp["n_estimators"], valid_sets=[dval],
        callbacks=[lgb.early_stopping(hp["early_stopping_rounds"], verbose=False)],
    )
    train_seconds = time.perf_counter() - t0
    prob = booster.predict(df[cols], num_iteration=booster.best_iteration)

    metrics = {
        "model": "lgbm", "groups": groups, "n_features": len(cols), "best_iteration": booster.best_iteration,
        "inner_valid_auc": float(roc_auc_score(val[TARGET], booster.predict(val[cols], num_iteration=booster.best_iteration))),
        "train_seconds": round(train_seconds, 2), "n_train_rows": len(fit), **_split_metrics(df, prob),
    }
    model_path = Path(cfg["paths"]["models_dir"]) / f"{name}.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    _save_outputs(cfg, name, df, prob, metrics, cols)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({**params, "n_estimators_max": hp["n_estimators"],
                           "early_stopping_rounds": hp["early_stopping_rounds"],
                           "groups": ",".join(groups), "dropped": ",".join(drop or []), "n_features": len(cols)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict({"features": cols}, "feature_list.json")
        if cfg.get("mlflow", {}).get("log_models", True):
            mlflow.lightgbm.log_model(booster, name="model")
    return {"name": name, **metrics}


def train_logreg(cfg: dict, groups: list[str] | None = None) -> dict:
    """Baseline: median impute + standard scale numerics, one-hot categoricals, balanced classes."""
    mlflow = mlflow_setup(cfg)
    groups = groups or ["app"]
    name = run_name("logreg", groups)
    cols = feature_columns(cfg, groups)
    df = load_frame(cfg, cols)
    cats = categorical_columns(df, cols)
    nums = [c for c in cols if c not in cats]
    X = df[cols].copy()
    X[cats] = X[cats].astype(object).where(X[cats].notna(), "missing")  # plain strings for sklearn

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
    ])
    hp = cfg["logreg"]
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=hp["C"], max_iter=hp["max_iter"], class_weight="balanced")),
    ])
    train_mask = (df["split"] == "train").to_numpy()
    t0 = time.perf_counter()
    model.fit(X[train_mask], df.loc[train_mask, TARGET])
    train_seconds = time.perf_counter() - t0
    prob = model.predict_proba(X)[:, 1]

    metrics = {"model": "logreg", "groups": groups, "n_features": len(cols),
               "n_encoded_features": int(model["pre"].transform(X.iloc[:5]).shape[1]),
               "train_seconds": round(train_seconds, 2), "n_train_rows": int(train_mask.sum()),
               **_split_metrics(df, prob)}
    model_path = Path(cfg["paths"]["models_dir"]) / f"{name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    _save_outputs(cfg, name, df, prob, metrics, cols)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({"C": hp["C"], "max_iter": hp["max_iter"], "class_weight": "balanced",
                           "impute": "median", "scale": "standard", "onehot_min_frequency": 20,
                           "groups": ",".join(groups), "n_features": len(cols)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict({"features": cols}, "feature_list.json")
        if cfg.get("mlflow", {}).get("log_models", True):
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype"])
    return {"name": name, **metrics}


def ablate(cfg: dict) -> list[dict]:
    """Cumulative table ablation: app, +bureau, +previous_application, +installments, +POS/credit card."""
    order = cfg["ablation"]
    rows, prev_auc, base_auc = [], None, None
    for i in range(len(order)):
        res = train_lgbm(cfg, order[: i + 1])
        auc = res["test_auc"]
        base_auc = auc if base_auc is None else base_auc
        rows.append({
            "step": i + 1, "added": order[i], "run": res["name"], "n_features": res["n_features"],
            "best_iteration": res["best_iteration"], "train_seconds": res["train_seconds"],
            "test_auc": round(auc, 5),
            "delta_vs_previous": None if prev_auc is None else round(auc - prev_auc, 5),
            "delta_vs_app_only": round(auc - base_auc, 5),
        })
        prev_auc = auc
    write_json(Path(cfg["paths"]["results_dir"]) / "ablation.json", rows)
    return rows
