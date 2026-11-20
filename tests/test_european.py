"""Stulz closed form: Monte Carlo agreement, symmetry, limits, no-arbitrage bounds."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from maxcall.european import bs_call, bvn_cdf, margrabe_max, mc_max_call, stulz_max_call

CASES = [
    # S1, S2, K, r, T, q1, q2, sigma1, sigma2, rho
    (100, 100, 100, 0.05, 3.0, 0.10, 0.10, 0.20, 0.20, 0.0),
    (100, 100, 100, 0.05, 1.0, 0.00, 0.00, 0.30, 0.20, 0.5),
    (90, 110, 100, 0.03, 2.0, 0.02, 0.05, 0.25, 0.35, -0.5),
    (100, 95, 105, 0.05, 0.5, 0.00, 0.03, 0.60, 0.80, 0.8),
    (120, 110, 100, 0.04, 1.5, 0.01, 0.00, 0.15, 0.40, 0.9),
]


@pytest.mark.parametrize("case", CASES)
def test_stulz_matches_monte_carlo(case):
    cf = stulz_max_call(*case)
    mc, se = mc_max_call(*case, n_paths=400_000, rng=np.random.default_rng(11))
    assert abs(mc - cf) < 3 * se


@pytest.mark.parametrize("case", CASES)
def test_stulz_symmetric_in_assets(case):
    S1, S2, K, r, T, q1, q2, s1, s2, rho = case
    a = stulz_max_call(S1, S2, K, r, T, q1, q2, s1, s2, rho)
    b = stulz_max_call(S2, S1, K, r, T, q2, q1, s2, s1, rho)
    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_stulz_between_single_asset_bounds(case):
    S1, S2, K, r, T, q1, q2, s1, s2, rho = case
    c1 = bs_call(S1, K, r, q1, s1, T)
    c2 = bs_call(S2, K, r, q2, s2, T)
    c = stulz_max_call(*case)
    assert max(c1, c2) - 1e-9 <= c <= c1 + c2 + 1e-9


def test_stulz_large_strike_is_worthless():
    assert stulz_max_call(100, 100, 1e5, 0.05, 1.0, 0.0, 0.0, 0.2, 0.3, 0.3) < 1e-10


def test_stulz_small_strike_tends_to_margrabe():
    args = (100, 90, 0.05, 1.0, 0.02, 0.01, 0.3, 0.2, 0.4)
    S1, S2, r, T, q1, q2, s1, s2, rho = args
    small_k = stulz_max_call(S1, S2, 1e-8, r, T, q1, q2, s1, s2, rho)
    assert small_k == pytest.approx(margrabe_max(S1, S2, T, q1, q2, s1, s2, rho), abs=1e-6)


def test_stulz_independent_low_vol_second_asset_close_to_bs():
    # Second asset far out of the money and nearly deterministic: price -> BS call on asset 1.
    c = stulz_max_call(100, 1.0, 100, 0.05, 1.0, 0.0, 0.0, 0.25, 0.01, 0.0)
    assert c == pytest.approx(bs_call(100, 100, 0.05, 0.0, 0.25, 1.0), abs=1e-8)


def test_stulz_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        stulz_max_call(100, 100, 0.0, 0.05, 1.0, 0, 0, 0.2, 0.2, 0.0)
    with pytest.raises(ValueError):
        stulz_max_call(100, 100, 100, 0.05, 1.0, 0, 0, 0.2, 0.2, 1.0)


@pytest.mark.parametrize("a,b,rho", [(0.0, 0.0, 0.0), (1.2, -0.3, -0.7), (-2.0, 1.5, 0.9),
                                     (0.3, 0.4, -0.99), (2.5, 3.0, 0.2)])
def test_bvn_cdf_matches_scipy(a, b, rho):
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([a, b])
    assert bvn_cdf(a, b, rho) == pytest.approx(ref, abs=1e-7)


def test_bvn_cdf_closed_form_at_origin():
    rho = 0.5
    assert bvn_cdf(0.0, 0.0, rho) == pytest.approx(0.25 + np.arcsin(rho) / (2 * np.pi), abs=1e-12)
