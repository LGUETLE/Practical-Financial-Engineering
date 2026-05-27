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