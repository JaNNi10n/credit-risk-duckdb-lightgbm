"""Whole pipeline on generated data: CLI stages, MLflow runs, report and the scoring API."""

import pytest
from fastapi.testclient import TestClient

from credit.__main__ import main
from credit.api import create_app
from credit.config import read_json
from credit.synthetic import write_tables
from tests.conftest import make_cfg


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    raw = write_tables(tmp / "raw", n_apps=4000, seed=3)
    cfg_path, cfg = make_cfg(tmp, raw, lgbm={"n_estimators": 300, "early_stopping_rounds": 30, "num_threads": 2},
                             explain={"n_rows": 500, "top_n": 20})
    main(["--config", str(cfg_path), "all", "--check-determinism"])
    return cfg


def test_all_result_files_written(e2e):
    res = e2e["paths"]["results_dir"]
    for name in ["features_summary.json", "splits_summary.json", "determinism.json", "ablation.json",
                 "calibration.json", "threshold.json", "explain.json", "stage_timings.json", "environment.json",
                 "runs/logreg_app.json", "runs/lgbm_all.json", "runs/lgbm_all_nosensitive.json"]:
        assert (res / name).exists(), name
    assert read_json(res / "determinism.json")["identical"]


def test_models_learn_signal(e2e):
    res = e2e["paths"]["results_dir"]
    assert read_json(res / "runs/logreg_app.json")["test_auc"] > 0.6
    assert read_json(res / "runs/lgbm_all.json")["test_auc"] > 0.6
    ablation = read_json(res / "ablation.json")
    assert [r["added"] for r in ablation] == e2e["ablation"]
    assert ablation[-1]["run"] == "lgbm_all"


def test_sensitive_columns_dropped(e2e):
    feats = read_json(e2e["paths"]["results_dir"] / "runs/lgbm_all_nosensitive.json")["features"]
    assert not set(e2e["sensitive_features"]) & set(feats)


def test_threshold_and_calibration_outputs(e2e):
    res = e2e["paths"]["results_dir"]
    thr = read_json(res / "threshold.json")
    assert 0.01 <= thr["main"]["optimal"]["threshold"] <= 0.99
    assert [s["lgd"] for s in thr["sensitivity"]] == [0.3, 0.45, 0.6]
    assert thr["main"]["approve_all"]["approval_rate"] == 1.0
    cal = read_json(res / "calibration.json")
    assert 0 < cal["test_brier_after"] < 0.25


def test_plots_and_report(e2e):
    reports = e2e["paths"]["reports_dir"]
    for name in ["reliability.png", "threshold_cost.png", "shap_beeswarm.png", "shap_waterfall_approved_1.png",
                 "shap_waterfall_rejected_2.png", "results.md"]:
        assert (reports / name).exists(), name
    text = (reports / "results.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in text
    assert "Table ablation" in text and "LGD sensitivity" in text


def test_mlflow_has_every_run(e2e):
    import mlflow

    from credit.train import EXPERIMENT, mlflow_setup

    mlflow_setup(e2e)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    names = set(runs["tags.mlflow.runName"])
    expected = {"logreg_app", "lgbm_app", "lgbm_all", "lgbm_all_nosensitive", "calibration_lgbm_all",
                "threshold_lgbm_all", "shap_lgbm_all"}
    assert expected <= names
    lgbm = runs[runs["tags.mlflow.runName"] == "lgbm_all"].iloc[0]
    assert lgbm["params.learning_rate"] == "0.05" and lgbm["metrics.test_auc"] > 0.6


def test_score_api(e2e):
    import pandas as pd

    client = TestClient(create_app(e2e))
    sid = int(pd.read_parquet(e2e["paths"]["features"], columns=["SK_ID_CURR"]).iloc[0, 0])
    r = client.post("/score", json={"sk_id_curr": sid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert len(body["top_reasons"]) == 5
    assert client.post("/score", json={"sk_id_curr": 999_999_999}).status_code == 404
    assert client.post("/score", json={"sk_id_curr": "abc"}).status_code == 422
    assert client.get("/health").json()["status"] == "ok"
