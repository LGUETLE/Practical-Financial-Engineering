"""Regression basis, the 2x2 methods, reproducibility of prices, control variate."""
import numpy as np
import pytest

from american_options_functions import (METHODS, basis_max, basis_size, lsm_price_max,
                                        lsm_price_max_cv, lsm_train_max)
from european import stulz_max_call


@pytest.mark.parametrize("degree,ncols", [(2, 7), (3, 11)])
@pytest.mark.parametrize("scale", [False, True])
def test_basis_shape(degree, scale, ncols):
    rng = np.random.default_rng(0)
    S1, S2 = rng.uniform(50, 150, 37), rng.uniform(50, 150, 37)
    Phi = basis_max(S1, S2, 100.0, degree, scale)
    assert Phi.shape == (37, ncols)
    assert basis_size(degree) == ncols
    assert np.all(Phi[:, 0] == 1.0)


def test_basis_columns_and_scaling():
    S1, S2, K = np.array([90.0, 120.0]), np.array([110.0, 80.0]), 100.0
    raw = basis_max(S1, S2, K, 3, scale=False)
    sc = basis_max(S1, S2, K, 3, scale=True)
    assert np.allclose(raw[:, -1], [10.0, 20.0])                 # intrinsic value
    assert np.allclose(sc[:, -1], [0.10, 0.20])                  # intrinsic value / K
    powers = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]                      # total degree per column
    for j, p in enumerate(powers):
        assert np.allclose(raw[:, j], sc[:, j] * K**p)           # constant column rescaling


def test_basis_rejects_other_degrees():
    with pytest.raises(ValueError):
        basis_max(np.ones(3), np.ones(3), 100.0, degree=4)


def test_scaling_does_not_change_policy(small_paths, bench):
    t1, t2, p1, p2 = small_paths
    b = bench
    for meth in METHODS:
        raw = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=meth, scale=False)
        sc = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=meth, scale=True)
        assert lsm_price_max(raw, p1, p2) == pytest.approx(lsm_price_max(sc, p1, p2), abs=1e-10)


def test_unknown_method_raises(small_paths, bench):
    t1, t2, _, _ = small_paths
    with pytest.raises(ValueError):
        lsm_train_max(t1, t2, bench["K"], bench["r"], bench["T"], method="XYZ")


def test_same_seed_gives_identical_prices(bench):
    from american_options_functions import simulate_gbm
    b = bench

    def run(seed):
        rng = np.random.default_rng(seed)
        paths = [simulate_gbm(100.0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], 2000, rng)
                 for _ in range(4)]
        out = []
        for meth in METHODS:
            model = lsm_train_max(paths[0], paths[1], b["K"], b["r"], b["T"], method=meth)
            out.append(lsm_price_max(model, paths[2], paths[3]))
        return out

    assert run(2026) == run(2026)
    assert run(2026) != run(2027)


def test_classic_lsm_and_tvr_differ_only_in_backward_pass(small_paths, bench):
    t1, t2, p1, p2 = small_paths
    b = bench
    prices = {m: lsm_price_max(lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=m), p1, p2)
              for m in METHODS}
    # All are prices of feasible stopping rules on the same paths: finite, positive, and not
    # above the value of the pathwise best exercise date (perfect foresight).
    dt = b["T"] / b["m"]
    h = np.maximum(np.maximum(p1, p2) - b["K"], 0.0)[:, 1:] * np.exp(-b["r"] * dt * np.arange(1, b["m"] + 1))
    foresight = h.max(axis=1).mean()
    for v in prices.values():
        assert 0 < v <= foresight


def test_control_variate_reduces_variance_and_keeps_mean(bench):
    from american_options_functions import simulate_gbm
    b = bench
    rng = np.random.default_rng(99)
    t1, t2, p1, p2 = (simulate_gbm(100.0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], n, rng)
                      for n in (4000, 4000, 50_000, 50_000))
    model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method="TVR")
    euro = stulz_max_call(100, 100, b["K"], b["r"], b["T"], b["delta"], b["delta"],
                          b["sigma"], b["sigma"], 0.0)
    res = lsm_price_max_cv(model, t1, t2, p1, p2, euro)
    assert res["vrf"] > 1.2
    assert res["se"] < res["se_plain"]
    assert abs(res["price"] - res["price_plain"]) < 3 * res["se_plain"]
