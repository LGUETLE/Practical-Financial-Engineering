"""
Closed-form prices for European options on two assets under correlated geometric Brownian motion
with continuous dividend yields (risk-neutral measure):

    dS_j = (r - q_j) S_j dt + sigma_j S_j dW_j,   d<W_1, W_2>_t = rho dt.

- stulz_max_call: European call on max(S_1, S_2) (Stulz 1982; Johnson 1987), exact.
- margrabe_max:   E[e^{-rT} max(S_1(T), S_2(T))], the K -> 0 limit of the max-call.
- bs_call:        Black-Scholes call with dividend yield (single asset).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import norm


def bvn_cdf(a: float, b: float, rho: float) -> float:
    """
    Standard bivariate normal CDF P(X <= a, Y <= b) with correlation rho, computed as
    integral_{-inf}^{a} phi(x) Phi((b - rho x) / sqrt(1 - rho^2)) dx by adaptive quadrature
    (absolute error ~1e-12). The degenerate cases |rho| = 1 are handled exactly.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [-1, 1], got {rho}")
    if np.isneginf(a) or np.isneginf(b):
        return 0.0
    if np.isposinf(a):
        return float(norm.cdf(b))
    if np.isposinf(b):
        return float(norm.cdf(a))
    if rho == 1.0:
        return float(norm.cdf(min(a, b)))
    if rho == -1.0:
        return float(max(norm.cdf(a) + norm.cdf(b) - 1.0, 0.0))
    s = np.sqrt(1.0 - rho * rho)
    f = lambda x: norm.pdf(x) * norm.cdf((b - rho * x) / s)
    # Integrate on a finite range: phi(x) < 1e-30 below -12.
    lo = -12.0
    if a <= lo:
        return 0.0
    val, _ = integrate.quad(f, lo, a, epsabs=1e-13, epsrel=1e-12, limit=200)
    return float(min(max(val, 0.0), 1.0))


def bs_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European call with continuous dividend yield q."""
    if K <= 0:
        return S * np.exp(-q * T) - K * np.exp(-r * T)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def margrabe_max(S1: float, S2: float, T: float, q1: float, q2: float,
                 sigma1: float, sigma2: float, rho: float) -> float:
    """
    Price of the claim paying max(S_1(T), S_2(T)) at T: S2 e^{-q2 T} plus Margrabe's (1978)
    option to exchange asset 2 for asset 1. Independent of r. Equals stulz_max_call with K = 0.
    """
    sig = np.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)
    sq = sig * np.sqrt(T)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sig**2) * T) / sq
    d2 = d1 - sq
    return float(S1 * np.exp(-q1 * T) * norm.cdf(d1) + S2 * np.exp(-q2 * T) * norm.cdf(-d2))


def stulz_max_call(S1: float, S2: float, K: float, r: float, T: float,
                   q1: float, q2: float, sigma1: float, sigma2: float, rho: float) -> float:
    """
    Exact price of a European call on the maximum of two assets, payoff
    max(max(S_1(T), S_2(T)) - K, 0), under correlated GBM with dividend yields q1, q2
    (Stulz 1982; Johnson 1987; notation as in Haug, The Complete Guide to Option Pricing
    Formulas):

      C = S1 e^{-q1 T} M(y1, d; rho1) + S2 e^{-q2 T} M(y2, -d + sigma sqrt(T); rho2)
          - K e^{-rT} [1 - M(-y1 + sigma1 sqrt(T), -y2 + sigma2 sqrt(T); rho)]

    with sigma^2 = sigma1^2 + sigma2^2 - 2 rho sigma1 sigma2,
    d  = [ln(S1/S2) + (q2 - q1 + sigma^2/2) T] / (sigma sqrt(T)),
    yj = [ln(Sj/K) + (r - qj + sigmaj^2/2) T] / (sigmaj sqrt(T)),
    rho1 = (sigma1 - rho sigma2) / sigma, rho2 = (sigma2 - rho sigma1) / sigma,
    and M the standard bivariate normal CDF. Requires K > 0 and sigma > 0
    (sigma = 0 means the two assets are identical up to scale).
    """
    if K <= 0:
        raise ValueError("K must be positive (use margrabe_max for K = 0)")
    sig2 = sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2
    if sig2 <= 0:
        raise ValueError("degenerate case sigma = 0 (perfectly correlated, equal volatilities)")
    sig = np.sqrt(sig2)
    sqT = np.sqrt(T)
    d = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sig2) * T) / (sig * sqT)
    y1 = (np.log(S1 / K) + (r - q1 + 0.5 * sigma1**2) * T) / (sigma1 * sqT)
    y2 = (np.log(S2 / K) + (r - q2 + 0.5 * sigma2**2) * T) / (sigma2 * sqT)
    rho1 = float(np.clip((sigma1 - rho * sigma2) / sig, -1.0, 1.0))
    rho2 = float(np.clip((sigma2 - rho * sigma1) / sig, -1.0, 1.0))
    term1 = S1 * np.exp(-q1 * T) * bvn_cdf(y1, d, rho1)
    term2 = S2 * np.exp(-q2 * T) * bvn_cdf(y2, -d + sig * sqT, rho2)
    term3 = K * np.exp(-r * T) * (1.0 - bvn_cdf(-y1 + sigma1 * sqT, -y2 + sigma2 * sqT, rho))
    return float(term1 + term2 - term3)


def mc_max_call(S1: float, S2: float, K: float, r: float, T: float,
                q1: float, q2: float, sigma1: float, sigma2: float, rho: float,
                n_paths: int, rng: np.random.Generator) -> tuple[float, float]:
    """
    Direct Monte Carlo price of the European max-call (exact terminal sampling, antithetic
    pairs not used). Unbiased; returns (price, standard error).
    """
    z1 = rng.standard_normal(n_paths)
    z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n_paths)
    ST1 = S1 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * z1)
    ST2 = S2 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * z2)
    pay = np.exp(-r * T) * np.maximum(np.maximum(ST1, ST2) - K, 0.0)
    return float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(n_paths))
