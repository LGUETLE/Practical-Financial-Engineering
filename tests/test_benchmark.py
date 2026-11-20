"""Slow end-to-end checks against the Glasserman (2003) Example 8.6.1 reference values."""
import numpy as np
import pytest

from maxcall.lsm import lsm_price_max, lsm_train_max
from maxcall.simulation import simulate_gbm
from maxcall.duality import duality_bounds


def _tvr_model_and_price(bench, S0, n_train, n_price, seed):
    b = bench
    rng = np.random.default_rng(seed)
    args = (S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"])
    t1, t2 = simulate_gbm(*args, n_train, rng), simulate_gbm(*args, n_train, rng)
    p1, p2 = simulate_gbm(*args, n_price, rng), simulate_gbm(*args, n_price, rng)
    model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method="TVR")
    return model, lsm_price_max(model, p1, p2, return_se=True)


@pytest.mark.slow
def test_benchmark_price_within_3_se_of_reference(bench):
    _, (price, se) = _tvr_model_and_price(bench, 100.0, 20_000, 100_000, seed=2026)
    assert abs(price - 13.90) < 3 * se
    assert price < 13.90 + 2 * se          # low-biased: not clearly above the reference


@pytest.mark.slow
def test_benchmark_reference_inside_duality_interval(bench):
    b = bench
    model, _ = _tvr_model_and_price(bench, 100.0, 20_000, 1000, seed=7)
    res = duality_bounds(model, 100.0, b["delta"], b["sigma"], b["sigma"], 0.0,
                         n_lower=100_000, n_outer=1000, n_inner=1000,
                         rng=np.random.default_rng(8))
    assert res["L"] <= res["U"]
    assert res["ci_low"] <= 13.90 <= res["ci_high"]
