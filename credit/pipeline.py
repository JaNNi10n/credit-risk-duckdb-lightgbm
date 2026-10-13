"""Feature pipeline: raw CSV -> DuckDB SQL (one file per table) -> one Parquet row per SK_ID_CURR."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from credit.config import SQL_DIR, read_json, write_json

# Kaggle file name for every raw table (the view name inside DuckDB is the dict key).
RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# features/<stem>.sql, in the column order of the final parquet. application must be first.
FEATURE_SQL = [
    "application",
    "bureau",
    "previous_application",
    "installments_payments",
    "pos_cash_balance",
    "credit_card_balance",
]
SECONDARY_SQL = FEATURE_SQL[1:]

# Table groups used by the ablation study (cumulative, in config order).
GROUPS = {
    "app": ["application"],
    "bureau": ["bureau"],
    "previous_application": ["previous_application"],
    "installments": ["installments_payments"],
    "pos_cc": ["pos_cash_balance", "credit_card_balance"],
}

ID, TARGET = "SK_ID_CURR", "TARGET"
SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"  # written by credit.synthetic; flags reports as not-real


def _q(path: Path) -> str:
    return "'" + Path(path).as_posix().replace("'", "''") + "'"


def connect(raw_dir: Path, threads: int = 8, memory_limit: str = "4GB", temp_dir: Path | None = None,
            agg_threads: int = 1) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one table per raw CSV and the past() macro used by the SQL files.

    The CSVs are parsed once, in parallel (`threads`). The feature queries then run on
    `agg_threads` threads: a parallel GROUP BY merges floating-point partial sums in an order that
    depends on thread scheduling, so with several threads the last bits of a SUM/AVG can change
    from run to run (seen on a 300k-applicant test build). One thread makes the output bit-identical.
    """
    raw_dir = Path(raw_dir)
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw files in {raw_dir}: {missing}. Run scripts/download.sh first "
            "(needs a Kaggle API token and the competition rules accepted)."
        )
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = {_q(temp_dir)}")
    con.execute("SET preserve_insertion_order = true")
    # A positive DAYS_* value lies after the application date (or is the 365243 sentinel).
    con.execute("CREATE MACRO past(x) AS CASE WHEN x <= 0 THEN x ELSE NULL END")
    for view, fname in RAW_FILES.items():
        # sample_size=-1: sniff types from the whole file, so a late float in an int-looking
        # column cannot break the load.
        con.execute(
            f"CREATE TABLE {view} AS SELECT * FROM read_csv({_q(raw_dir / fname)}, header = true, sample_size = -1)"
        )
    con.execute(f"SET threads = {int(agg_threads)}")
    return con


def run_sql(con: duckdb.DuckDBPyConnection, stem: str) -> str:
    """Materialise features/<stem>.sql as table f_<stem>; returns the table name."""
    sql = (SQL_DIR / f"{stem}.sql").read_text(encoding="utf-8")
    con.execute(f"CREATE OR REPLACE TABLE f_{stem} AS {sql}")
    return f"f_{stem}"


def _columns(con, table: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_features(cfg: dict) -> dict:
    """Run every SQL file, LEFT JOIN onto application, write data/features.parquet + summary."""
    paths = cfg["paths"]
    out = Path(paths["features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    db = cfg["duckdb"]
    con = connect(paths["raw_dir"], db["threads"], db["memory_limit"], db.get("temp_directory"), db.get("agg_threads", 1))
    load_seconds = round(time.perf_counter() - t_start, 2)

    tables, groups = {}, {}
    for stem in FEATURE_SQL:
        t0 = time.perf_counter()
        name = run_sql(con, stem)
        cols = [c for c in _columns(con, name) if c not in (ID, TARGET)]
        rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        dup = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT {ID}) FROM {name}").fetchone()[0]
        if dup:
            raise ValueError(f"{name} has {dup} duplicate {ID} rows")
        if stem != "application" and TARGET in _columns(con, name):
            raise ValueError(f"{name} must not carry {TARGET}")
        tables[stem] = {"features": len(cols), "rows": rows, "seconds": round(time.perf_counter() - t0, 2)}
        groups[stem] = cols

    # Count-like columns become 0 (not NULL) for applicants with no rows in that table.
    select = ["a.*"]
    for stem in SECONDARY_SQL:
        for c in groups[stem]:
            expr = f"COALESCE({c}, 0)" if ("count" in c or c.endswith("nunique")) else c
            select.append(f"{expr} AS {c}")
    joins = " ".join(f"LEFT JOIN f_{s} USING ({ID})" for s in SECONDARY_SQL)
    query = f"SELECT {', '.join(select)} FROM f_application a {joins} ORDER BY {ID}"

    t0 = time.perf_counter()
    con.execute(f"COPY ({query}) TO {_q(out)} (FORMAT parquet, COMPRESSION zstd)")
    write_seconds = round(time.perf_counter() - t0, 2)

    n_app = con.execute("SELECT COUNT(*) FROM application").fetchone()[0]
    n_out, n_ids = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {ID}) FROM read_parquet({_q(out)})").fetchone()
    con.close()
    if not (n_out == n_ids == n_app):
        raise ValueError(f"row mismatch: application={n_app} output={n_out} distinct ids={n_ids}")

    write_json(Path(paths["feature_groups"]), groups)
    summary = {
        "rows": n_out,
        "synthetic": (Path(paths["raw_dir"]) / SYNTHETIC_MARKER).exists(),
        "n_features_total": sum(t["features"] for t in tables.values()),
        "tables": tables,
        "csv_load_seconds": load_seconds,
        "join_and_write_seconds": write_seconds,
        "total_seconds": round(time.perf_counter() - t_start, 2),
        "duckdb_threads": cfg["duckdb"]["threads"],
        "duckdb_agg_threads": cfg["duckdb"].get("agg_threads", 1),
        "duckdb_version": duckdb.__version__,
        "sha256": file_sha256(out),
    }
    write_json(Path(paths["results_dir"]) / "features_summary.json", summary)
    return summary


def check_determinism(cfg: dict, first_hash: str) -> dict:
    """Rebuild the feature store into a temp folder and compare the Parquet SHA-256 with the first build."""
    import copy
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg2 = copy.deepcopy(cfg)
        cfg2["paths"].update(features=Path(tmp) / "features.parquet", feature_groups=Path(tmp) / "groups.json",
                             results_dir=Path(tmp))
        second = build_features(cfg2)["sha256"]
    res = {"runs": 2, "hashes": [first_hash, second], "identical": first_hash == second}
    write_json(Path(cfg["paths"]["results_dir"]) / "determinism.json", res)
    return res


def make_splits(cfg: dict) -> pd.DataFrame:
    """Stratified 70/10/20 train/calib/test split of application_train IDs (seed from config)."""
    paths, frac, seed = cfg["paths"], cfg["split"], cfg["seed"]
    df = duckdb.sql(f"SELECT {ID}, {TARGET} FROM read_parquet({_q(paths['features'])}) ORDER BY {ID}").df()
    rest, test = train_test_split(df, test_size=frac["test"], stratify=df[TARGET], random_state=seed)
    calib_share = frac["calib"] / (frac["train"] + frac["calib"])
    train, calib = train_test_split(rest, test_size=calib_share, stratify=rest[TARGET], random_state=seed)
    splits = pd.concat(
        [train.assign(split="train"), calib.assign(split="calib"), test.assign(split="test")]
    ).sort_values(ID)[[ID, "split"]].reset_index(drop=True)
    Path(paths["splits"]).parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(paths["splits"], index=False)

    stats = df.merge(splits, on=ID).groupby("split")[TARGET].agg(["size", "mean"])
    write_json(
        Path(paths["results_dir"]) / "splits_summary.json",
        {s: {"rows": int(r["size"]), "default_rate": round(float(r["mean"]), 5)} for s, r in stats.iterrows()},
    )
    return splits


def feature_columns(cfg: dict, groups: list[str], drop: list[str] | None = None) -> list[str]:
    """Feature names for a list of ablation groups ('all' = every group)."""
    table_cols = read_json(cfg["paths"]["feature_groups"])
    if groups == ["all"]:
        groups = list(GROUPS)
    cols = [c for g in groups for stem in GROUPS[g] for c in table_cols[stem]]
    return [c for c in cols if c not in set(drop or [])]
