"""Optional scoring API: POST /score {"sk_id_curr": 100002} -> probability, decision, top-5 SHAP reasons.

Applicants are looked up in the feature store (data/features.parquet); unknown IDs return 404.
Run: uvicorn credit.api:create_app --factory --port 8000   (after `python -m credit all`).
Set CREDIT_CONFIG to use a config.yaml other than the repo default.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit.config import load_config, read_json
from credit.pipeline import ID


class ScoreRequest(BaseModel):
    sk_id_curr: int = Field(..., gt=0, description="SK_ID_CURR present in the feature store")


class Reason(BaseModel):
    feature: str
    value: float | str | None
    shap_logodds: float


class ScoreResponse(BaseModel):
    sk_id_curr: int
    probability: float = Field(description="isotonic-calibrated P(default)")
    raw_probability: float
    threshold: float
    decision: str
    top_reasons: list[Reason]


def create_app(cfg: dict | None = None, run: str = "lgbm_all") -> FastAPI:
    cfg = cfg or load_config(os.environ.get("CREDIT_CONFIG"))
    paths = cfg["paths"]
    booster = lgb.Booster(model_file=str(Path(paths["models_dir"]) / f"{run}.txt"))
    iso = joblib.load(Path(paths["models_dir"]) / f"isotonic_{run}.joblib")
    threshold = float(read_json(Path(paths["results_dir"]) / "threshold.json")["main"]["optimal"]["threshold"])
    cols = booster.feature_name()
    features_path = Path(paths["features"]).as_posix()
    con = duckdb.connect()  # one cursor per request below: DuckDB connections are not thread-safe

    app = FastAPI(title="Home Credit default-risk scorer", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": run, "threshold": threshold, "n_features": len(cols)}

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        row = con.cursor().execute(f"SELECT * FROM read_parquet(?) WHERE {ID} = ?", [features_path, req.sk_id_curr]).df()
        if row.empty:
            raise HTTPException(status_code=404, detail=f"SK_ID_CURR {req.sk_id_curr} not found in feature store")
        X = row[cols].copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category")  # LightGBM re-maps to its training categories
        raw = float(booster.predict(X)[0])
        prob = float(iso.predict([raw])[0])
        contrib = booster.predict(X, pred_contrib=True)[0][:-1]  # TreeSHAP values, last entry is the bias
        top = np.argsort(-np.abs(contrib))[:5]
        reasons = []
        for i in top:
            v = X.iloc[0, i]
            v = None if pd.isna(v) else (str(v) if isinstance(v, str) else float(v))
            reasons.append(Reason(feature=cols[i], value=v, shap_logodds=round(float(contrib[i]), 5)))
        return ScoreResponse(sk_id_curr=req.sk_id_curr, probability=round(prob, 6), raw_probability=round(raw, 6),
                             threshold=threshold, decision="reject" if prob >= threshold else "approve",
                             top_reasons=reasons)

    return app

