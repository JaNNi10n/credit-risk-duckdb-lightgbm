"""Rebuild the feature store several times and compare Parquet SHA-256 hashes.

Used to check the determinism fix: the current pipeline (CSVs loaded once into tables, GROUP BY
on `agg_threads`) against the same queries with 8 GROUP BY threads, and against the pre-fix
layout where every query read the CSVs through views.

    python scripts/compare_agg_threads.py CONFIG AGG_THREADS {tables,views} N_BUILDS OUT_DIR

In views mode the queries always run on `duckdb.threads` (as before the fix), so AGG_THREADS is ignored.
"""

import copy
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import credit.pipeline as pl  # noqa: E402
from credit.config import load_config  # noqa: E402


def connect_views(raw_dir, threads=8, memory_limit="4GB", temp_dir=None, agg_threads=1):
    """Pre-fix layout: one view per CSV, so each query re-parses its CSVs with `threads` threads."""
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = true")
    con.execute("CREATE MACRO past(x) AS CASE WHEN x <= 0 THEN x ELSE NULL END")
    for view, fname in pl.RAW_FILES.items():
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_csv({pl._q(Path(raw_dir) / fname)}, header = true, sample_size = -1)")
    return con


def main():
    cfg_path, agg, mode, n, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
    if mode == "views":
        pl.connect = connect_views
    cfg = load_config(cfg_path)
    cfg["duckdb"]["agg_threads"] = agg
    hashes = []
    for i in range(n):
        c = copy.deepcopy(cfg)
        d = out_dir / f"{mode}_agg{agg}_{i}"
        c["paths"].update(features=d / "features.parquet", feature_groups=d / "groups.json", results_dir=d)
        t0 = time.perf_counter()
        sha = pl.build_features(c)["sha256"]
        hashes.append(sha)
        print(f"mode={mode} agg_threads={agg} build={i} seconds={time.perf_counter() - t0:.1f} sha256={sha}", flush=True)
    print(f"mode={mode} agg_threads={agg} distinct_hashes={len(set(hashes))} of {n} builds")


if __name__ == "__main__":
    main()
