# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-13

### Added

- DuckDB feature store: one SQL file per Home Credit table in `features/`, each grouped by `SK_ID_CURR`. `bureau_balance` is aggregated to `SK_ID_BUREAU` first. `credit/pipeline.py` LEFT JOINs everything onto `application_train` and writes `data/features.parquet`, with the feature count per table.
- Stratified 70/10/20 train/calibration/test split (seed 42), saved to `data/splits.parquet`.
- Logistic Regression baseline (application features only) and LightGBM with early stopping. Every run logs params, metrics, the feature list and the model to a local MLflow store.
- Cumulative table ablation (app, +bureau, +previous_application, +installments, +POS/credit card).
- Isotonic calibration fitted on the calibration split, with Brier score, ECE and a 10-bin reliability diagram.
- SHAP TreeExplainer: a global beeswarm and local waterfalls for approved and rejected applicants.
- Cost-matrix threshold (`config.yaml: cost`) chosen on the calibration split, scored on test against the 0.5 cut-off and approve-all, with an LGD sensitivity table.
- Fairness check: LightGBM retrained without `CODE_GENDER`, `DAYS_BIRTH` and `APP_EMPLOYED_TO_AGE`.
- `python -m credit report` writes `reports/results.md` from the files in `results/`.
- Optional FastAPI `POST /score` with probability, decision and top-5 SHAP reasons (404 for unknown IDs).
- pytest suite for leakage, data contracts, determinism, the cost maths, the full CLI and the API, all on generated fixture data. GitHub Actions runs ruff and pytest.
- `credit/synthetic.py`: generated tables with the real Home Credit column names, for tests and CI only.

### Fixed

- The feature Parquet was not bit-identical between two builds of a 300k-applicant generated dataset, because DuckDB merged parallel floating-point partial sums in a varying order. CSVs are now loaded once in parallel and the GROUP BY queries run on one thread (`duckdb.agg_threads`).

### Not done yet

- No run on the real Kaggle data, because no Kaggle API token was available on the build machine. The README therefore reports no model-quality numbers.
