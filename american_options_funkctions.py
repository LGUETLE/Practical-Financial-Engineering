"""
==============================================================================
American Max-Option Pricing – Longstaff-Schwartz (LSM)
Based on Glasserman (2003), Chapter 8.6, Example 8.6.1
==============================================================================
"""

import numpy as np
from scipy.linalg import lstsq

# ─────────────────────────────────────────────────────────────────────────────
#  1. GBM SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_gbm(S0: float, sigma: float, r: float, delta: float,
                 T: float, m: int, n_paths: int, seed: int = None) -> np.ndarray:
    """
    Simulates paths of a Geometric Brownian Motion (GBM).
    Incorporates a continuous dividend yield (delta) as specified in 
    Glasserman, Example 8.6.1.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / m
    # Drift includes the dividend yield (r - delta)
    drift = (r - delta - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    Z = np.random.standard_normal((n_paths, m))
    log_S = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    
    return np.hstack([np.full((n_paths, 1), S0), np.exp(log_S)])

# ─────────────────────────────────────────────────────────────────────────────
#  2. BASIS FUNCTIONS FOR THE MAX-OPTION (Table 8.1)
# ─────────────────────────────────────────────────────────────────────────────

def basis_max(S1: np.ndarray, S2: np.ndarray, K: float) -> np.ndarray:
    """
    Basis functions exactly matching Glasserman Table 8.1 (last row, Low = 13.78).
    Includes: 1, S_i, S_i^2, S1*S2, h(S1, S2).
    Manual scaling has been removed to prevent data leakage between train/price sets.
    """
    h = np.maximum(np.maximum(S1, S2) - K, 0.0)
    
    # Create the design matrix without artificial scaling
    cols = [
        np.ones(len(S1)),       # 1
        S1, S2,                 # S_i
        S1**2, S2**2,           # S_i^2
        S1 * S2,                # S1 * S2
        h                       # h(S1, S2)
    ]
    return np.column_stack(cols)
# ─────────────────────────────────────────────────────────────────────────────
#  3. LSM ALGORITHM (Out-of-Sample / "Low Estimator")
# ─────────────────────────────────────────────────────────────────────────────
def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float) -> dict:
    """
    Backward induction to estimate the regression coefficients (betas).
    Uses Longstaff-Schwartz logic over ALL paths (as Glasserman noted for this example).
    """
    m = train1.shape[1] - 1
    dt = T / m
    df_step = np.exp(-r * dt)

    # V tracks the realized, discounted cash flows
    V = np.maximum(np.maximum(train1[:, -1], train2[:, -1]) - K, 0.0)
    betas = {}

    for i in range(m - 1, 0, -1):
        V *= df_step
        
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i = np.maximum(np.maximum(S1_i, S2_i) - K, 0.0)
        
        Phi = basis_max(S1_i, S2_i, K)
        
        # Regression over all paths
        beta = lstsq(Phi, V)[0]
        betas[i] = beta

        # Expected continuation value (capped at 0 per Glasserman p. 464)
        C_hat = np.maximum(Phi @ beta, 0.0)
        
        # Exercise decision: intrinsic value must strictly exceed continuation value
        exercise = h_i > C_hat
        
        # Update the value vector for paths where early exercise is optimal
        V[exercise] = h_i[exercise]

    return dict(betas=betas, K=K, r=r, T=T, m=m)

def lsm_price_max(model: dict, price1: np.ndarray, price2: np.ndarray) -> float:
    """
    Forward pass on independent paths to calculate the true Out-of-Sample price.
    Corresponds to the 'Low Estimator' from Glasserman.
    """
    betas = model["betas"]
    K, r, T, m = model["K"], model["r"], model["T"], model["m"]
    df_step = np.exp(-r * (T / m))

    n = price1.shape[0]
    exercised = np.zeros(n, dtype=bool)
    payoffs = np.zeros(n)
    ex_times = np.full(n, m, dtype=int)

    # Forward iteration to find the FIRST optimal stopping time
    for i in range(1, m):
        active = ~exercised
        if not np.any(active): 
            break

        S1_i = price1[active, i]
        S2_i = price2[active, i]
        h_i = np.maximum(np.maximum(S1_i, S2_i) - K, 0.0)

        Phi = basis_max(S1_i, S2_i, K)
        C_hat = np.maximum(Phi @ betas[i], 0.0)
        
        ex = h_i > C_hat

        # Record the payoff and exercise time for paths that stop here
        idx = np.where(active)[0]
        payoffs[idx[ex]] = h_i[ex]
        ex_times[idx[ex]] = i
        exercised[idx[ex]] = True

    # Terminal condition for paths that never exercised early
    not_ex = ~exercised
    payoffs[not_ex] = np.maximum(np.maximum(price1[not_ex, -1], price2[not_ex, -1]) - K, 0.0)
    ex_times[not_ex] = m

    # Discount all cash flows back to t=0
    price = float(np.mean(payoffs * df_step ** ex_times))
    return price

# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION (Parameters from Glasserman Example 8.6.1)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Parameters matching Glasserman Example 8.6.1
    S0 = 100.0        # Initial price S1(0) = S2(0)
    K_val = 100.0     # Strike K
    T_val = 3.0       # Time to maturity (years)
    r_val = 0.05      # Risk-free interest rate (5%)
    delta_val = 0.10  # Dividend yield (10%)
    sigma_val = 0.20  # Volatility (20%)
    m_steps = 9       # Number of exercise opportunities
    n_paths = 4000    # 4000 paths for regression, 4000 for pricing

    print("Starting Low-Estimator Simulation for American Max-Option...")

    # 1. Generate paths for the training phase (Asset 1 and 2)
    train_S1 = simulate_gbm(S0, sigma_val, r_val, delta_val, T_val, m_steps, n_paths, seed=42)
    train_S2 = simulate_gbm(S0, sigma_val, r_val, delta_val, T_val, m_steps, n_paths, seed=43)

    # 2. Generate new, independent paths for the pricing phase (Low Estimator)
    price_S1 = simulate_gbm(S0, sigma_val, r_val, delta_val, T_val, m_steps, n_paths, seed=44)
    price_S2 = simulate_gbm(S0, sigma_val, r_val, delta_val, T_val, m_steps, n_paths, seed=45)

    # 3. Training: Estimate regression coefficients
    model_max = lsm_train_max(train_S1, train_S2, K_val, r_val, T_val)
    
    # 4. Pricing: Apply out-of-sample forward pass
    price_max = lsm_price_max(model_max, price_S1, price_S2)
    
    print("-" * 60)
    print(f"Estimated Price (LSM Low Estimator): {price_max:.2f}")
    print("Glasserman Benchmark (Table 8.1): Exact price is 13.90")
    print("-" * 60)