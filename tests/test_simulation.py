"""GBM simulation: risk-neutral mean, correlation, volatility, reproducibility."""
import numpy as np
import pytest

from maxcall.simulation import simulate_correlated_gbm, simulate_gbm


@pytest.mark.parametrize("S0,sigma,r,delta,T", [(100.0, 0.2, 0.05, 0.10, 3.0),
                                                 (50.0, 0.6, 0.03, 0.0, 1.0)])
def test_gbm_risk_neutral_mean(S0, sigma, r, delta, T):
    n = 200_000
    S = simulate_gbm(S0, sigma, r, delta, T, 9, n, np.random.default_rng(1))
    assert S.shape == (n, 10)
    assert np.all(S[:, 0] == S0)
    ST = S[:, -1]
    target = S0 * np.exp((r - delta) * T)
    se = ST.std(ddof=1) / np.sqrt(n)
    assert abs(ST.mean() - target) < 4 * se


def test_correlated_gbm_mean_correlation_and_volatility():
    n, m, T = 200_000, 12, 1.0
    s1, s2, rho, r, delta = 0.3, 0.5, 0.6, 0.05, 0.02
    P1, P2 = simulate_correlated_gbm(100.0, 80.0, s1, s2, rho, r, delta, T, m, n,
                                     np.random.default_rng(2))
    for P, S0, s in ((P1, 100.0, s1), (P2, 80.0, s2)):
        ST = P[:, -1]
        se = ST.std(ddof=1) / np.sqrt(n)
        assert abs(ST.mean() - S0 * np.exp((r - delta) * T)) < 4 * se
        lr = np.diff(np.log(P), axis=1).ravel()
        assert abs(lr.std() * np.sqrt(m / T) - s) < 0.005
    lr1 = np.diff(np.log(P1), axis=1).ravel()
    lr2 = np.diff(np.log(P2), axis=1).ravel()
    assert abs(np.corrcoef(lr1, lr2)[0, 1] - rho) < 0.005


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.9])
def test_correlated_gbm_log_return_correlation(rho):
    P1, P2 = simulate_correlated_gbm(100.0, 100.0, 0.4, 0.4, rho, 0.05, 0.0, 1.0, 4, 100_000,
                                     np.random.default_rng(3))
    c = np.corrcoef(np.diff(np.log(P1), axis=1).ravel(), np.diff(np.log(P2), axis=1).ravel())[0, 1]
    assert abs(c - rho) < 0.01


def test_simulation_is_reproducible():
    a = simulate_gbm(100.0, 0.2, 0.05, 0.1, 3.0, 9, 1000, np.random.default_rng(7))
    b = simulate_gbm(100.0, 0.2, 0.05, 0.1, 3.0, 9, 1000, np.random.default_rng(7))
    c = simulate_gbm(100.0, 0.2, 0.05, 0.1, 3.0, 9, 1000, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
