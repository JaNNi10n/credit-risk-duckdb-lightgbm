import numpy as np
import pytest

from credit.calibrate import ece, reliability
from credit.threshold import best_threshold, total_cost

Y = np.array([1, 0, 0, 1, 0])
P = np.array([0.9, 0.1, 0.2, 0.05, 0.6])
A = np.array([100.0, 200.0, 300.0, 400.0, 500.0])


def test_approve_all_costs_lgd_times_defaulted_credit():
    assert total_cost(Y, P, A, t=1.01, lgd=0.45, fp_margin=0.1) == pytest.approx(0.45 * (100 + 400))


def test_reject_all_costs_margin_on_good_clients():
    assert total_cost(Y, P, A, t=0.0, lgd=0.45, fp_margin=0.1) == pytest.approx(0.1 * (200 + 300 + 500))


def test_mixed_decision_cost():
    # t=0.5: approve rows 1,2,3 (p<0.5) -> row 3 is a missed default; reject rows 0,4 -> row 4 is a lost good client
    assert total_cost(Y, P, A, t=0.5, lgd=0.45, fp_margin=0.1) == pytest.approx(0.45 * 400 + 0.1 * 500)


def test_perfect_scores_reach_zero_cost():
    y = np.array([0, 0, 1, 1])
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    t = best_threshold(y, np.array([0.01, 0.02, 0.98, 0.99]), np.ones(4), 0.45, 0.1, grid)
    assert total_cost(y, np.array([0.01, 0.02, 0.98, 0.99]), np.ones(4), t, 0.45, 0.1) == 0


def test_calibrated_scores_recover_bayes_threshold():
    """With calibrated probabilities the cost-optimal cut-off is margin / (LGD + margin)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    t = best_threshold(y, p, np.ones(p.size), 0.45, 0.10, grid)
    assert abs(t - 0.10 / 0.55) <= 0.02


def test_reliability_and_ece():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    table = reliability(y, p, n_bins=10)
    assert list(table["n"]) == [2, 2]
    assert ece(table) == pytest.approx(0.05)
