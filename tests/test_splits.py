import pandas as pd

from credit.pipeline import ID, TARGET, make_splits


def test_split_ids_are_disjoint_and_cover_everything(built):
    cfg, _ = built
    s = pd.read_parquet(cfg["paths"]["splits"])
    ids = {k: set(g[ID]) for k, g in s.groupby("split")}
    assert set(ids) == {"train", "calib", "test"}
    assert not ids["train"] & ids["calib"]
    assert not ids["train"] & ids["test"]
    assert not ids["calib"] & ids["test"]
    features = pd.read_parquet(cfg["paths"]["features"], columns=[ID])
    assert set.union(*ids.values()) == set(features[ID])


def test_split_proportions_and_stratification(built):
    cfg, _ = built
    s = pd.read_parquet(cfg["paths"]["splits"]).merge(pd.read_parquet(cfg["paths"]["features"], columns=[ID, TARGET]))
    share = s["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.01
    assert abs(share["calib"] - 0.10) < 0.01
    assert abs(share["test"] - 0.20) < 0.01
    rates = s.groupby("split")[TARGET].mean()
    assert (rates - s[TARGET].mean()).abs().max() < 0.03


def test_splits_are_reproducible(built):
    cfg, _ = built
    first = pd.read_parquet(cfg["paths"]["splits"])
    again = make_splits(cfg)
    pd.testing.assert_frame_equal(first, again)
