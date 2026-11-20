"""
Exact simulation of geometric Brownian motions under the risk-neutral measure on the exercise grid
t_i = i T / m. All randomness comes from an explicitly passed np.random.Generator.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


def simulate_gbm(S0: float, sigma: float, r: float, delta: float,
                 T: float, m: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Simulates paths of a geometric Brownian motion under the risk-neutral measure,
    dS = (r - delta) S dt + sigma S dW, exactly on the grid t_i = i T / m.
    Used for the theoretical Glasserman benchmark.

    All randomness comes from the explicitly passed generator `rng`
    (np.random.default_rng(seed)); no global NumPy state is used.
    Returns an array of shape (n_paths, m + 1) whose first column is S0.
    """
    dt = T / m
    drift = (r - delta - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, m))
    log_S = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    
    return np.hstack([np.full((n_paths, 1), S0), np.exp(log_S)])

def simulate_correlated_gbm(S0_1: float, S0_2: float, sigma_1: float, sigma_2: float,
                            rho: float, r: float, delta: float, T: float, m: int,
                            n_paths: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates two correlated GBMs under the risk-neutral measure,
    dS_j = (r - delta) S_j dt + sigma_j S_j dW_j with d<W_1, W_2> = rho dt,
    exactly on the grid t_i = i T / m (correlation via Cholesky factor of [[1, rho], [rho, 1]]).
    Used for the calibrated (empirical) scenarios: only sigma_1, sigma_2 and rho are calibrated
    from data; the drift is always the risk-neutral r - delta, never a historical estimate.

    All randomness comes from the explicitly passed generator `rng`.
    Returns two arrays of shape (n_paths, m + 1) whose first columns are S0_1 and S0_2.
    """
    dt = T / m
    L = cholesky(np.array([[1.0, rho], [rho, 1.0]]), lower=True)
    Z = rng.standard_normal((2, n_paths, m))
    Z1 = L[0, 0] * Z[0]
    Z2 = L[1, 0] * Z[0] + L[1, 1] * Z[1]
    sq = np.sqrt(dt)
    log1 = np.cumsum((r - delta - 0.5 * sigma_1**2) * dt + sigma_1 * sq * Z1, axis=1)
    log2 = np.cumsum((r - delta - 0.5 * sigma_2**2) * dt + sigma_2 * sq * Z2, axis=1)
    paths_1 = np.hstack([np.full((n_paths, 1), S0_1), S0_1 * np.exp(log1)])
    paths_2 = np.hstack([np.full((n_paths, 1), S0_2), S0_2 * np.exp(log2)])
    return paths_1, paths_2
