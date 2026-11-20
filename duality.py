"""
Primal-dual upper bound for a trained exercise policy (Andersen & Broadie 2004;
Glasserman 2003, Sec. 8.7), for the Bermudan max-call on two GBMs.

Notation (all values discounted to t = 0, exercise dates i = 1, ..., m):
  h_i   discounted payoff at date i,
  L_i   value of following the policy from date i on (h_i if the policy exercises at i),
  Q_i   = E_i[L_{i+1}], continuation value of the policy at date i,
  M_i   = sum_{k < i} (L_{k+1} - Q_k), a martingale with M_0 = 0.
For any martingale M with M_0 = 0, V_0 <= E[max_i (h_i - M_i)] (dual representation). The max can
be restricted to the dates with h_i > 0 and maturity, since stopping with h_i = 0 is never better
than waiting until maturity (h_m >= 0). Telescoping gives

  h_i - M_i = Q_0 + D_i,
  D_i = h_i - L_i - sum_{k < i, policy exercises at k} (h_k - Q_k),

so U = Q_0 + E[max_i D_i] and the duality gap is Delta = E[max_i D_i] >= 0 (D = 0 at the first
exercise date). The Q_k are estimated by inner simulation, unbiased given the outer path; they
enter M linearly, so by Jensen's inequality the estimated bound is still high-biased for any
number of inner paths (inner noise only widens the gap). Q_0 is the policy value itself; it is
estimated by the lower-bound estimator L on independent paths, so U = L + Delta.
"""
from __future__ import annotations

import time

import numpy as np

from american_options_functions import (apply_policy, continuation_value,
                                        discounted_policy_payoffs)


def _simulate_from(S1: np.ndarray, S2: np.ndarray, n_steps: int, dt: float, r: float,
                   delta: float, sigma1: float, sigma2: float, rho: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact GBM continuation of n paths from states (S1, S2) over n_steps steps of size dt."""
    n = len(S1)
    Z1 = rng.standard_normal((n, n_steps))
    Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal((n, n_steps))
    sq = np.sqrt(dt)
    inc1 = (r - delta - 0.5 * sigma1**2) * dt + sigma1 * sq * Z1
    inc2 = (r - delta - 0.5 * sigma2**2) * dt + sigma2 * sq * Z2
    P1 = S1[:, None] * np.exp(np.cumsum(inc1, axis=1))
    P2 = S2[:, None] * np.exp(np.cumsum(inc2, axis=1))
    return P1, P2


def simulate_outer(S0: float, n: int, T: float, m: int, r: float, delta: float,
                   sigma1: float, sigma2: float, rho: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Outer paths of shape (n, m + 1) starting at S0 for both assets."""
    start = np.full(n, float(S0))
    P1, P2 = _simulate_from(start, start, m, T / m, r, delta, sigma1, sigma2, rho, rng)
    return np.hstack([start[:, None], P1]), np.hstack([start[:, None], P2])


def inner_continuation_values(model: dict, S1_k: np.ndarray, S2_k: np.ndarray, k: int,
                              n_inner: int, delta: float, sigma1: float, sigma2: float,
                              rho: float, rng: np.random.Generator,
                              max_rows: int = 400_000) -> np.ndarray:
    """
    Unbiased inner-simulation estimates of Q_k = E_k[discounted payoff of the policy started at
    date k + 1] for each state (S1_k, S2_k): n_inner continuations per state, policy applied from
    date k + 1, payoffs discounted to t = 0. Processed in chunks of at most max_rows paths.
    """
    m, T, r = model["m"], model["T"], model["r"]
    dt = T / m
    n_states = len(S1_k)
    out = np.empty(n_states)
    per_chunk = max(1, max_rows // n_inner)
    for c0 in range(0, n_states, per_chunk):
        c1 = min(n_states, c0 + per_chunk)
        s1 = np.repeat(S1_k[c0:c1], n_inner)
        s2 = np.repeat(S2_k[c0:c1], n_inner)
        P1, P2 = _simulate_from(s1, s2, m - k, dt, r, delta, sigma1, sigma2, rho, rng)
        full1 = np.zeros((len(s1), m + 1))
        full2 = np.zeros((len(s1), m + 1))
        full1[:, k] = s1
        full2[:, k] = s2
        full1[:, k + 1:] = P1
        full2[:, k + 1:] = P2
        payoff, ex_idx = apply_policy(model, full1, full2, first=k + 1)
        disc = payoff * np.exp(-r * dt * ex_idx)
        out[c0:c1] = disc.reshape(c1 - c0, n_inner).mean(axis=1)
    return out


def pathwise_gap(h: np.ndarray, Q: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """
    Pathwise duality gap max_i D_i (see module docstring) from discounted payoffs h (n, m+1),
    continuation-value estimates Q (n, m; only needed where h > 0 at dates 1..m-1) and the
    policy's exercise decisions ex (n, m+1; ex[:, m] is True). The max runs over the dates
    with h_i > 0 and over maturity. The result is >= 0 on every path.
    """
    n, m1 = h.shape
    m = m1 - 1
    D = np.full((n, m + 1), -np.inf)
    running = np.zeros(n)                                    # sum over exercise dates k < i
    for i in range(1, m):
        itm = h[:, i] > 0
        L_i = np.where(ex[:, i], h[:, i], Q[:, i])
        D[itm, i] = h[itm, i] - L_i[itm] - running[itm]
        e = itm & ex[:, i]
        running[e] += h[e, i] - Q[e, i]
    D[:, m] = -running                                       # h_m - L_m = 0 at maturity
    return D.max(axis=1)


def duality_bounds(model: dict, S0: float, delta: float, sigma1: float, sigma2: float,
                   rho: float, n_lower: int, n_outer: int, n_inner: int,
                   rng: np.random.Generator) -> dict:
    """
    Lower and upper bound for the Bermudan max-call price under a trained policy.

    Lower bound L: forward pass of the policy on n_lower independent paths (low-biased).
    Upper bound U = L + Delta: Andersen-Broadie duality gap Delta estimated on n_outer outer
    paths with n_inner inner paths per continuation value (high-biased). L and Delta use
    independent random numbers, so SE_U^2 = SE_L^2 + SE_Delta^2.
    Returns L, SE_L, U, SE_U, gap, SE_gap, the 95% interval [L - 1.96 SE_L, U + 1.96 SE_U],
    the number of inner simulations, the runtime and the pathwise arrays ("lower_paths",
    "gap_paths") for paired comparisons of policies on common random numbers.
    """
    t0 = time.perf_counter()
    K, T, r, m = model["K"], model["T"], model["r"], model["m"]
    dt = T / m

    # Lower bound
    L1, L2 = simulate_outer(S0, n_lower, T, m, r, delta, sigma1, sigma2, rho, rng)
    y = discounted_policy_payoffs(model, L1, L2)
    L, se_L = float(y.mean()), float(y.std(ddof=1) / np.sqrt(n_lower))

    # Outer paths for the gap
    O1, O2 = simulate_outer(S0, n_outer, T, m, r, delta, sigma1, sigma2, rho, rng)
    disc = np.exp(-r * dt * np.arange(m + 1))
    h = np.maximum(np.maximum(O1, O2) - K, 0.0) * disc       # discounted payoffs, (n, m+1)

    Q = np.full((n_outer, m), np.nan)                        # Q_k for k = 1..m-1 where ITM
    ex = np.zeros((n_outer, m + 1), dtype=bool)              # policy decision at date k
    n_inner_sims = 0
    for k in range(1, m):
        itm = h[:, k] > 0
        idx = np.flatnonzero(itm)
        if idx.size == 0:
            continue
        Q[idx, k] = inner_continuation_values(model, O1[idx, k], O2[idx, k], k, n_inner,
                                              delta, sigma1, sigma2, rho, rng)
        n_inner_sims += idx.size
        C_hat = continuation_value(model, k, O1[idx, k], O2[idx, k]) * disc[k]
        ex[idx, k] = h[idx, k] > C_hat
    ex[:, m] = True

    gap_path = pathwise_gap(h, Q, ex)
    gap = float(gap_path.mean())
    se_gap = float(gap_path.std(ddof=1) / np.sqrt(n_outer))

    U = L + gap
    se_U = float(np.sqrt(se_L**2 + se_gap**2))
    return dict(L=L, se_L=se_L, U=U, se_U=se_U, gap=gap, se_gap=se_gap,
                ci_low=L - 1.96 * se_L, ci_high=U + 1.96 * se_U,
                n_lower=n_lower, n_outer=n_outer, n_inner=n_inner,
                inner_estimates=n_inner_sims, min_gap_path=float(gap_path.min()),
                seconds=time.perf_counter() - t0, lower_paths=y, gap_paths=gap_path)
