"""Duality upper bound: telescoped gap formula, non-negativity, L <= U."""
import numpy as np

from american_options_functions import lsm_train_max
from duality import duality_bounds, pathwise_gap


def _brute_force_gap(h, Q, ex):
    """max over {h > 0} U {m} of h_i - M_i with the untelescoped martingale (Q_0 = 0)."""
    n, m1 = h.shape
    m = m1 - 1
    L = np.where(ex, h, np.hstack([Q, np.zeros((n, 1))]))
    L[:, m] = h[:, m]
    Qfull = np.hstack([np.zeros((n, 1)), Q[:, 1:]])
    M = np.zeros((n, m + 1))
    for i in range(1, m + 1):
        M[:, i] = M[:, i - 1] + L[:, i] - Qfull[:, i - 1]
    cand = np.where((h > 0) | (np.arange(m + 1) == m), h - M, -np.inf)
    cand[:, 0] = -np.inf
    return cand.max(axis=1)


def test_pathwise_gap_equals_brute_force_martingale():
    rng = np.random.default_rng(0)
    n, m = 500, 9
    h = np.maximum(rng.normal(0, 3, (n, m + 1)), 0.0)
    h[:, 0] = 0
    Q = rng.uniform(0, 5, (n, m))
    ex = (h > rng.uniform(0, 5, (n, m + 1))) & (h > 0)
    ex[:, m] = True
    gap = pathwise_gap(h, Q, ex)
    assert np.allclose(gap, _brute_force_gap(h, Q, ex), atol=1e-12)
    assert np.all(gap >= 0)
    # Continuation estimates at out-of-the-money dates must not matter
    Q2 = Q.copy()
    Q2[h[:, :m] == 0] = 1e6
    assert np.array_equal(pathwise_gap(h, Q2, ex), gap)


def test_lower_bound_not_above_upper_bound(small_paths, bench):
    t1, t2, _, _ = small_paths
    b = bench
    model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method="TVR")
    res = duality_bounds(model, 100.0, b["delta"], b["sigma"], b["sigma"], 0.0,
                         n_lower=5000, n_outer=200, n_inner=100, rng=np.random.default_rng(4))
    assert res["L"] <= res["U"]
    assert res["gap"] >= 0
    assert res["min_gap_path"] >= 0
    assert res["ci_low"] < res["L"] <= res["U"] < res["ci_high"]
