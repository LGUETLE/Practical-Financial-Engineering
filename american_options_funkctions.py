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

def basis_max(S1: np.ndarray, S2: np.ndarray, K: float,
              s1_scale: float = None, s2_scale: float = None) -> tuple[np.ndarray, float, float]:
    """
    Generates basis functions for the max option.
    Uses the final specification from Glasserman Table 8.1:
    1, Si, Si^2, S1*S2, and the intrinsic value h(S1, S2).
    """
    # Calculate scaling factors to avoid numerical instability in the regression
    if s1_scale is None:
        s1_scale = float(S1.mean())
    if s2_scale is None:
        s2_scale = float(S2.mean())

    s1 = S1 / s1_scale
    s2 = S2 / s2_scale
    
    # Intrinsic value of the option: h(S1, S2) = (max(S1, S2) - K)^+
    h = np.maximum(np.maximum(S1, S2) - K, 0.0)
    h_scale = max(h.mean(), 1e-8)
    h_norm = h / h_scale

    # Design matrix (Si implies both S1 and S2 are included)
    cols = [
        np.ones(len(S1)),  # Constant 1
        s1, s2,            # Linear terms
        s1**2, s2**2,      # Quadratic terms
        s1 * s2,           # Interaction term
        h_norm             # Intrinsic value
    ]

    return np.column_stack(cols), s1_scale, s2_scale

# ─────────────────────────────────────────────────────────────────────────────
#  3. LSM ALGORITHM (Out-of-Sample / "Low Estimator")
# ─────────────────────────────────────────────────────────────────────────────

def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float) -> dict:
    """
    Rückwärts-Induktion zur Schätzung der Regressionskoeffizienten (beta).
    Kompakte Variante mit 'Rolling Present Value' (V), um Double-Discounting zu vermeiden.
    """
    n, m1 = train1.shape
    m = m1 - 1
    dt = T / m
    df_step = np.exp(-r * dt)

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K, 0.0)

    # V speichert den realisierten, rollierenden Barwert der zukünftigen Cashflows
    V = payoff(train1[:, -1], train2[:, -1])

    betas, scales1, scales2 = {}, {}, {}

    # Rückwärts durch die Zeit (i = m-1, ..., 1)
    for i in range(m - 1, 0, -1):
        # 1. Zukünftigen Wert auf den aktuellen Zeitpunkt i abdiskontieren
        V *= df_step
        
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i = payoff(S1_i, S2_i)
        
        # Nur in-the-money Pfade betrachten
        itm = h_i > 0

        if itm.sum() < 10: 
            continue

        # 2. Regression (V[itm] ist bereits korrekt auf i abdiskontiert!)
        pv = V[itm]
        Phi, sc1, sc2 = basis_max(S1_i[itm], S2_i[itm], K)
        beta, _, _, _ = lstsq(Phi, pv)

        betas[i] = beta
        scales1[i] = sc1
        scales2[i] = sc2

        # 3. Fortsetzungswert schätzen
        C_hat = np.maximum(Phi @ beta, 0.0)
        
        # 4. Strikte Ungleichung nach Glasserman (S. 464)
        exercise = h_i[itm] > C_hat
        
        # 5. Bei optimaler Ausübung den abdiskontierten Wert durch den sofortigen Payoff ersetzen
        idx = np.where(itm)[0]
        V[idx[exercise]] = h_i[itm][exercise]

    return dict(betas=betas, scales1=scales1, scales2=scales2, K=K, r=r, T=T, m=m)


def lsm_price_max(model: dict, price1: np.ndarray, price2: np.ndarray) -> float:
    """
    Forward pass on a new set of independent paths to calculate the option price.
    This corresponds to the "Low Estimator" in Glasserman (8.33)-(8.34).
    """
    betas = model["betas"]
    sc1_map, sc2_map = model["scales1"], model["scales2"]
    K_, r_, T_, m = model["K"], model["r"], model["T"], model["m"]
    df_step = np.exp(-r_ * (T_ / m))

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K_, 0.0)

    n = price1.shape[0]
    exercised = np.zeros(n, dtype=bool)
    payoffs = np.zeros(n)
    ex_times = np.full(n, m, dtype=int)

    # Forward iteration to find the first stopping time where h_i >= C_hat
    for i in range(1, m):
        if i not in betas: 
            continue
            
        active = ~exercised
        if active.sum() == 0: 
            break

        S1_i = price1[active, i]
        S2_i = price2[active, i]
        h_i = payoff(S1_i, S2_i)

        # Apply scaling factors computed during the training phase
        Phi, _, _ = basis_max(S1_i, S2_i, K_, s1_scale=sc1_map[i], s2_scale=sc2_map[i])
        
        C_hat = np.maximum(Phi @ betas[i], 0.0)
        ex = h_i > C_hat

        idx = np.where(active)[0]
        payoffs[idx[ex]] = h_i[ex]
        ex_times[idx[ex]] = i
        exercised[idx[ex]] = True

    # For paths that were never exercised, use the terminal payoff
    not_ex = ~exercised
    payoffs[not_ex] = payoff(price1[not_ex, -1], price2[not_ex, -1])
    ex_times[not_ex] = m

    # Discount all cashflows back to t=0
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