import numpy as np
from scipy.linalg import lstsq, cholesky

# ─────────────────────────────────────────────────────────────────────────────
#  1. GBM SIMULATION (Independent for Theory & Correlated for Empirics)
# ─────────────────────────────────────────────────────────────────────────────

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

def simulate_correlated_gbm(S0_1: float, S0_2: float, mu_1: float, mu_2: float, 
                            sigma_1: float, sigma_2: float, rho: float,
                            T: float, m: int, n_paths: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates two correlated GBMs using Cholesky decomposition.
    Used for historical market data.

    All randomness comes from the explicitly passed generator `rng`.
    """
    dt = T / m
    # Covariance matrix for Cholesky decomposition
    cov_matrix = np.array([[1.0, rho], 
                           [rho, 1.0]])
    L = cholesky(cov_matrix, lower=True)

    # Generate independent shocks and apply correlation
    Z_uncorr = rng.standard_normal((2, n_paths, m))
    
    paths_1 = np.zeros((n_paths, m + 1))
    paths_2 = np.zeros((n_paths, m + 1))
    paths_1[:, 0] = S0_1
    paths_2[:, 0] = S0_2

    for t in range(1, m + 1):
        Z_corr = L @ Z_uncorr[:, :, t-1] 
        paths_1[:, t] = paths_1[:, t-1] * np.exp((mu_1 - 0.5 * sigma_1**2) * dt + sigma_1 * np.sqrt(dt) * Z_corr[0, :])
        paths_2[:, t] = paths_2[:, t-1] * np.exp((mu_2 - 0.5 * sigma_2**2) * dt + sigma_2 * np.sqrt(dt) * Z_corr[1, :])
        
    return paths_1, paths_2

# ─────────────────────────────────────────────────────────────────────────────
#  2. BASIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def basis_max(S1: np.ndarray, S2: np.ndarray, K: float) -> np.ndarray:
    """
    Regression basis for the continuation value of the max-call: 7 terms.

    Complete degree-2 polynomial in (S1, S2) -- 1, S1, S2, S1^2, S2^2, S1*S2 --
    plus the intrinsic value max(max(S1, S2) - K, 0). The polynomials are not
    scaled; K is used only for the intrinsic-value term.
    """
    h = np.maximum(np.maximum(S1, S2) - K, 0.0)
    cols = [
        np.ones(len(S1)),       # 1 (Constant)
        S1, S2,                 # Linear terms
        S1**2, S2**2,           # Quadratic terms
        S1 * S2,                # Interaction term
        h                       # Intrinsic value
    ]
    return np.column_stack(cols)

# ─────────────────────────────────────────────────────────────────────────────
#  3. LSM / TVR ALGORITHM (Backward & Forward)
# ─────────────────────────────────────────────────────────────────────────────

def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float, method: str = "TVR") -> dict:
    """
    Backward induction phase. 
    Supports Classic LSM (ITM filter) and TVR (Value Iteration).
    """
    m = train1.shape[1] - 1
    dt = T / m
    df_step = np.exp(-r * dt)

    # Terminal condition
    V = np.maximum(np.maximum(train1[:, -1], train2[:, -1]) - K, 0.0)
    betas = {}

    for i in range(m - 1, 0, -1):
        Y = V * df_step  # Discounted future values
        
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i = np.maximum(np.maximum(S1_i, S2_i) - K, 0.0)
        
        Phi = basis_max(S1_i, S2_i, K)
        
        if method == "LSM":
            # 1. Classic LSM: Regress only over In-The-Money paths
            itm = h_i > 0
            if np.sum(itm) > 0:
                beta = lstsq(Phi[itm], Y[itm])[0]
            else:
                beta = np.zeros(Phi.shape[1])
            betas[i] = beta

            # 2. Cashflow Tracking: Track exact stopping time
            C_hat = np.zeros_like(h_i)
            C_hat[itm] = np.maximum(Phi[itm] @ beta, 0.0)
            exercise = h_i > C_hat
            V = np.where(exercise, h_i, Y) 
            
        elif method == "TVR":
            # 1. TVR: Regress over ALL paths
            beta = lstsq(Phi, Y)[0]
            betas[i] = beta
            
            # 2. Value Iteration: Smooth the target variable
            C_hat = np.maximum(Phi @ beta, 0.0)
            V = np.maximum(h_i, C_hat)

    return dict(betas=betas, K=K, r=r, T=T, m=m, method=method)

def lsm_price_max(model: dict, price1: np.ndarray, price2: np.ndarray,
                  return_se: bool = False) -> float | tuple[float, float]:
    """
    Forward pass (Out-of-Sample Pricing).
    The method is identical for LSM and TVR since the decision rule
    only relies on the trained beta coefficients.

    The exercise policy is learned on separate training paths and applied to
    the independent pricing paths given here, so the estimator is low-biased
    (E[price] <= true Bermudan price).

    If return_se is True, also returns the standard error of the estimate:
    sample std (ddof=1) of the discounted pathwise payoffs divided by sqrt(n).
    It reflects the noise of the pricing paths only, conditional on the
    trained policy (training-path noise is not included).
    """
    betas = model["betas"]
    K, r, T, m = model["K"], model["r"], model["T"], model["m"]
    df_step = np.exp(-r * (T / m))

    n = price1.shape[0]
    exercised = np.zeros(n, dtype=bool)
    payoffs = np.zeros(n)
    ex_times = np.full(n, m, dtype=int)

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

        # Record payoff and exercise time
        idx = np.where(active)[0]
        payoffs[idx[ex]] = h_i[ex]
        ex_times[idx[ex]] = i
        exercised[idx[ex]] = True

    # Terminal condition for paths that reached maturity
    not_ex = ~exercised
    payoffs[not_ex] = np.maximum(np.maximum(price1[not_ex, -1], price2[not_ex, -1]) - K, 0.0)
    ex_times[not_ex] = m

    # Discount all cash flows to t=0
    discounted = payoffs * df_step ** ex_times
    price = float(np.mean(discounted))
    if return_se:
        se = float(np.std(discounted, ddof=1) / np.sqrt(n))
        return price, se
    return price