"""Leakage and data-contract tests for the DuckDB feature pipeline (fixture data only)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from credit.config import SQL_DIR, read_json
from credit.pipeline import (
    FEATURE_SQL,
    ID,
    RAW_FILES,
    SECONDARY_SQL,
    TARGET,
    build_features,
    check_determinism,
    connect,
    run_sql,
)
from tests.conftest import FIXTURES, make_cfg

SECONDARY_RAW = [v for v in RAW_FILES if v != "application"]


def _app():
    return pd.read_csv(FIXTURES / "application_train.csv")


def _features(cfg):
    return pd.read_parquet(cfg["paths"]["features"])


def test_exactly_one_row_per_applicant(built):
    cfg, _ = built
    df = _features(cfg)
    assert df[ID].is_unique
    assert set(df[ID]) == set(_app()[ID])


def test_output_rows_equal_application_rows(built):
    cfg, summary = built
    assert len(_features(cfg)) == len(_app()) == summary["rows"]


def test_secondary_tables_contain_ids_not_in_application():
    """Sanity check on the fixtures: the LEFT JOIN must drop orphan IDs, as in the real data."""
    bureau = pd.read_csv(FIXTURES / "bureau.csv")
    assert not set(bureau[ID]).issubset(set(_app()[ID]))


def test_no_target_in_any_secondary_aggregate():
    con = connect(FIXTURES, threads=2)
    for stem in SECONDARY_SQL:
        sql = (SQL_DIR / f"{stem}.sql").read_text(encoding="utf-8")
        assert "TARGET" not in sql.upper()
        assert "FROM APPLICATION" not in " ".join(sql.upper().split())
        cols = [r[0] for r in con.execute(f"DESCRIBE {run_sql(con, stem)}").fetchall()]
        assert TARGET not in cols, stem
        assert cols[0] == ID


def test_every_sql_file_is_one_row_per_sk_id_curr():
    con = connect(FIXTURES, threads=2)
    for stem in FEATURE_SQL:
        t = run_sql(con, stem)
        n, n_distinct = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {ID}) FROM {t}").fetchone()
        assert n == n_distinct, stem


def _secondary_tables(raw_dir):
    con = connect(raw_dir, threads=2)
    return {s: con.execute(f"SELECT * FROM {run_sql(con, s)} ORDER BY {ID}").df() for s in SECONDARY_SQL}


def test_positive_days_values_are_never_used(tmp_path):
    """Rewrite every positive (future) DAYS_* value in the secondary tables to another positive
    value. If no feature uses future dates, every secondary aggregate is unchanged."""
    mutated = tmp_path / "raw"
    shutil.copytree(FIXTURES, mutated)
    n_positive = 0
    for view in SECONDARY_RAW:
        path = mutated / RAW_FILES[view]
        df = pd.read_csv(path)
        for c in [c for c in df.columns if c.startswith("DAYS_")]:
            pos = df[c] > 0
            n_positive += int(pos.sum())
            df.loc[pos, c] = df.loc[pos, c] * 7 + 99_999
        df.to_csv(path, index=False)
    assert n_positive > 50, "fixtures must contain future DAYS_* values for this test to mean anything"

    original, changed = _secondary_tables(FIXTURES), _secondary_tables(mutated)
    for stem in SECONDARY_SQL:
        pd.testing.assert_frame_equal(original[stem], changed[stem], check_exact=False, rtol=1e-12, obj=stem)


def test_secondary_days_features_are_never_positive(built):
    cfg, _ = built
    df = _features(cfg)
    groups = read_json(cfg["paths"]["feature_groups"])
    days_cols = [c for s in SECONDARY_SQL for c in groups[s] if "days" in c and "late" not in c and "early" not in c]
    assert days_cols
    assert (df[days_cols].max() <= 0).all()


def test_days_employed_sentinel_becomes_null_with_flag(built):
    cfg, _ = built
    raw = _app()[[ID, "DAYS_EMPLOYED"]]
    out = _features(cfg)[[ID, "DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM"]].merge(raw, on=ID, suffixes=("", "_raw"))
    sentinel = out["DAYS_EMPLOYED_raw"] == 365243
    assert sentinel.sum() > 0
    assert out.loc[sentinel, "DAYS_EMPLOYED"].isna().all()
    assert (out.loc[sentinel, "DAYS_EMPLOYED_ANOM"] == 1).all()
    assert out.loc[~sentinel, "DAYS_EMPLOYED"].notna().all()
    assert (out.loc[~sentinel, "DAYS_EMPLOYED_ANOM"] == 0).all()
    assert not (_features(cfg).select_dtypes("number") == 365243).any().any()


def test_no_infinite_values(built):
    cfg, _ = built
    num = _features(cfg).select_dtypes("number")
    assert np.isfinite(num.fillna(0).to_numpy()).all()


def test_applicants_without_history_get_zero_counts(built):
    cfg, _ = built
    df = _features(cfg)
    bureau_ids = set(pd.read_csv(FIXTURES / "bureau.csv")[ID])
    no_bureau = ~df[ID].isin(bureau_ids)
    assert no_bureau.sum() > 0
    assert (df.loc[no_bureau, "bur_count"] == 0).all()
    assert df.loc[no_bureau, "bur_amt_credit_sum"].isna().all()


def test_aggregates_match_pandas(built):
    """Spot-check SQL aggregates against an independent pandas computation."""
    cfg, _ = built
    df = _features(cfg).set_index(ID)
    prev = pd.read_csv(FIXTURES / "previous_application.csv")
    refused = prev.assign(r=prev["NAME_CONTRACT_STATUS"].eq("Refused")).groupby(ID)["r"].mean()
    common = refused.index.intersection(df.index)
    np.testing.assert_allclose(df.loc[common, "prev_refused_share"], refused[common])

    inst = pd.read_csv(FIXTURES / "installments_payments.csv")
    ok = (inst["DAYS_INSTALMENT"] <= 0) & (inst["DAYS_ENTRY_PAYMENT"] <= 0)
    late = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).where(ok)
    rate = (late > 0).groupby(inst[ID]).mean()
    common = rate.index.intersection(df.index)
    np.testing.assert_allclose(df.loc[common, "inst_late_rate"], rate[common])

    app = _app().set_index(ID)
    np.testing.assert_allclose(df["APP_CREDIT_TO_INCOME"], app.loc[df.index, "AMT_CREDIT"] / app.loc[df.index, "AMT_INCOME_TOTAL"])


def test_feature_count_per_table_is_recorded(built):
    cfg, summary = built
    groups = read_json(cfg["paths"]["feature_groups"])
    assert set(summary["tables"]) == set(FEATURE_SQL)
    for stem, info in summary["tables"].items():
        assert info["features"] == len(groups[stem]) > 0
    assert summary["n_features_total"] == len(_features(cfg).columns) - 2  # minus SK_ID_CURR, TARGET


def test_pipeline_is_deterministic(built):
    cfg, summary = built
    det = check_determinism(cfg, summary["sha256"])
    assert det["identical"], det


def test_missing_raw_files_give_a_clear_error(tmp_path):
    _, cfg = make_cfg(tmp_path, tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="download.sh"):
        build_features(cfg)


def test_feature_queries_run_single_threaded():
    """Parallel GROUP BYs merge float partial sums in a scheduling-dependent order; a 300k-row
    generated build gave different parquet hashes with 8 threads. The SQL must run on agg_threads."""
    con = connect(FIXTURES, threads=4)
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 1
    assert con.execute("SELECT current_setting('preserve_insertion_order')").fetchone()[0] is True
