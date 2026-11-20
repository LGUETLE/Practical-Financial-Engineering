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

# Default regression basis, chosen by the ablation in theory_engine.py (run_basis_ablation):
# degree 2 gave the higher mean (low-biased) price than degree 3 at N = 4,000 for both LSM and
# TVR; scaling by K does not change the price but lowers the condition number by ~3-5 orders.
DEFAULT_DEGREE = 2
DEFAULT_SCALE = True


def basis_max(S1: np.ndarray, S2: np.ndarray, K: float,
              degree: int = DEFAULT_DEGREE, scale: bool = DEFAULT_SCALE) -> np.ndarray:
    """
    Regression basis for the continuation value of the max-call.

    The basis is a complete polynomial of total degree `degree` in (x1, x2) plus
    the intrinsic value of the option:
      degree 2:  1, x1, x2, x1^2, x2^2, x1*x2, h                       ->  7 columns
      degree 3:  the six above + x1^3, x1^2*x2, x1*x2^2, x2^3, h       -> 11 columns

    If `scale` is True, the basis is evaluated on x_j = S_j / K and h = max(max(x1, x2) - 1, 0),
    i.e. the intrinsic value divided by K. If False, x_j = S_j and h is the raw intrinsic value.
    Scaling uses only the fixed contract strike K, which is known before any path is
    simulated, so it uses no information from the training or pricing paths (no data leakage).
    Scaling by a constant is an invertible linear change of the columns: in exact arithmetic it
    leaves the least-squares fit, the exercise policy and the price unchanged. It only reduces
    the condition number of the regression matrix (S^3 ~ 1e6 next to the constant 1 when raw).
    """
    if degree not in (2, 3):
        raise ValueError(f"degree must be 2 or 3, got {degree}")
    if scale:
        x1, x2 = S1 / K, S2 / K
        h = np.maximum(np.maximum(x1, x2) - 1.0, 0.0)
    else:
        x1, x2 = S1, S2
        h = np.maximum(np.maximum(S1, S2) - K, 0.0)
    cols = [np.ones(len(x1)), x1, x2, x1**2, x2**2, x1 * x2]
    if degree == 3:
        cols += [x1**3, x1**2 * x2, x1 * x2**2, x2**3]
    cols.append(h)
    return np.column_stack(cols)


def basis_size(degree: int) -> int:
    """Number of columns of basis_max for the given polynomial degree (7 or 11)."""
    return {2: 7, 3: 11}[degree]


def continuation_value(model: dict, i: int, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Fitted continuation value max(phi(S)' beta_i, 0) of a trained model at exercise date i."""
    Phi = basis_max(S1, S2, model["K"], model["degree"], model["scale"])
    return np.maximum(Phi @ model["betas"][i], 0.0)

# ─────────────────────────────────────────────────────────────────────────────
#  3. LSM / TVR ALGORITHM (Backward & Forward)
# ─────────────────────────────────────────────────────────────────────────────

def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float, method: str = "TVR",
                  degree: int = DEFAULT_DEGREE, scale: bool = DEFAULT_SCALE,
                  diagnostics: bool = False) -> dict:
    """
    Backward induction on training paths: estimates regression coefficients beta_i of the
    continuation value at the exercise dates i = 1, ..., m-1 (date m is maturity).

    Methods:
      "LSM"  classic Longstaff-Schwartz: regress the realized discounted cash flow of the
             current policy on the basis, in-the-money paths only; propagated values are
             realized cash flows (regression error affects only the exercise decision).
      "TVR"  Tsitsiklis-Van Roy: regress on all paths with target max(h_{i+1}, C_hat_{i+1})
             (regression-based value iteration); fitted values are propagated.

    The coefficients define an exercise policy; the price is obtained out of sample with
    lsm_price_max (low-biased). With diagnostics=True the model also contains the 2-norm
    condition number of the regression matrix at each date ("cond").
    """
    m = train1.shape[1] - 1
    dt = T / m
    df_step = np.exp(-r * dt)

    # Terminal condition
    V = np.maximum(np.maximum(train1[:, -1], train2[:, -1]) - K, 0.0)
    betas = {}
    cond = {}

    for i in range(m - 1, 0, -1):
        Y = V * df_step  # Discounted future values
        
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i = np.maximum(np.maximum(S1_i, S2_i) - K, 0.0)
        
        Phi = basis_max(S1_i, S2_i, K, degree, scale)
        
        if method == "LSM":
            # 1. Classic LSM: Regress only over In-The-Money paths
            itm = h_i > 0
            if np.sum(itm) > 0:
                beta = lstsq(Phi[itm], Y[itm])[0]
                if diagnostics:
                    cond[i] = float(np.linalg.cond(Phi[itm]))
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
            if diagnostics:
                cond[i] = float(np.linalg.cond(Phi))
            
            # 2. Value Iteration: Smooth the target variable
            C_hat = np.maximum(Phi @ beta, 0.0)
            V = np.maximum(h_i, C_hat)
        else:
            raise ValueError(f"unknown method {method!r}")

    model = dict(betas=betas, K=K, r=r, T=T, m=m, method=method,
                 degree=degree, scale=scale)
    if diagnostics:
        model["cond"] = cond
    return model

def lsm_price_max(model: dict, price1: np.ndarray, price2: np.ndarray,
                  return_se: bool = False) -> float | tuple[float, float]:
    """
    Forward pass (Out-of-Sample Pricing).
    The method is identical for LSM and TVR since the decision rule
    only relies on the trained beta coefficients: exercise at date i if
    h_i > max(phi(S_i)' beta_i, 0), otherwise at maturity.

    The exercise policy is learned on separate training paths and applied to
    the independent pricing paths given here, so the estimator is low-biased
    (E[price] <= true Bermudan price).

    If return_se is True, also returns the standard error of the estimate:
    sample std (ddof=1) of the discounted pathwise payoffs divided by sqrt(n).
    It reflects the noise of the pricing paths only, conditional on the
    trained policy (training-path noise is not included).
    """
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

        C_hat = continuation_value(model, i, S1_i, S2_i)
        
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
