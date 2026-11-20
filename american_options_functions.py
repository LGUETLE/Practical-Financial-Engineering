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

# 2x2 design: regression target x regression sample.
METHODS: dict[str, tuple[str, str]] = {
    "LSM":     ("cashflow", "itm"),   # classic Longstaff-Schwartz
    "LSM-all": ("cashflow", "all"),
    "TVR-ITM": ("value", "itm"),
    "TVR":     ("value", "all"),      # Tsitsiklis-Van Roy
}


def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float, method: str = "TVR",
                  degree: int = DEFAULT_DEGREE, scale: bool = DEFAULT_SCALE,
                  diagnostics: bool = False) -> dict:
    """
    Backward induction on training paths: estimates regression coefficients beta_i of the
    continuation value at the exercise dates i = 1, ..., m-1 (date m is maturity).

    Model: Bermudan call on max(S1, S2) with payoff h = max(max(S1, S2) - K, 0); paths are
    simulated under the risk-neutral measure, discounting at rate r.

    The method is one cell of a 2x2 design (see METHODS):
      target  "cashflow": regression target is the realized discounted cash flow Y of the
                          policy estimated at later dates; after the exercise decision the
                          path keeps h_i (exercise) or Y (continue). Regression errors affect
                          only the decisions.
              "value":    regression target is the discounted value estimate
                          V_{i+1} = max(h_{i+1}, C_hat_{i+1}) (regression-based value
                          iteration); fitted values are propagated, so errors can accumulate.
      sample  "itm":      fit on in-the-money paths (h_i > 0) only.
              "all":      fit on all paths.
    "LSM" = (cashflow, itm) is classic Longstaff-Schwartz, "TVR" = (value, all) is
    Tsitsiklis-Van Roy. For (value, itm) the continuation value of out-of-the-money paths is
    the fitted function evaluated outside the region it was fitted on (extrapolation).

    Estimator: the coefficients define an exercise policy; the price is obtained out of sample
    with lsm_price_max on independent paths and is therefore low-biased for every method.
    With diagnostics=True the model also contains the 2-norm condition number of the
    regression matrix at each date ("cond").
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    target, sample = METHODS[method]
    m = train1.shape[1] - 1
    dt = T / m
    df_step = np.exp(-r * dt)

    # Terminal condition
    V = np.maximum(np.maximum(train1[:, -1], train2[:, -1]) - K, 0.0)
    betas = {}
    cond = {}

    for i in range(m - 1, 0, -1):
        Y = V * df_step  # discounted value / cash flow from date i+1
        
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i = np.maximum(np.maximum(S1_i, S2_i) - K, 0.0)
        Phi = basis_max(S1_i, S2_i, K, degree, scale)

        rows = h_i > 0 if sample == "itm" else np.ones(len(h_i), dtype=bool)
        if np.any(rows):
            beta = lstsq(Phi[rows], Y[rows])[0]
            if diagnostics:
                cond[i] = float(np.linalg.cond(Phi[rows]))
        else:
            beta = np.zeros(Phi.shape[1])
        betas[i] = beta

        C_hat = np.maximum(Phi @ beta, 0.0)
        if target == "cashflow":
            # Exercise only if h_i > C_hat (never when out of the money, h_i = 0)
            V = np.where(h_i > C_hat, h_i, Y)
        else:
            # Value iteration: propagate the fitted value
            V = np.maximum(h_i, C_hat)

    model = dict(betas=betas, K=K, r=r, T=T, m=m, method=method,
                 degree=degree, scale=scale)
    if diagnostics:
        model["cond"] = cond
    return model

def apply_policy(model: dict, S1: np.ndarray, S2: np.ndarray,
                 first: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the trained stopping rule to paths S1, S2 of shape (n, m + 1) from exercise date
    `first` on: stop at the first date i in {first, ..., m-1} with h_i > max(phi(S_i)' beta_i, 0),
    otherwise at maturity m. Columns before `first` are ignored.
    Returns (undiscounted payoff at the stopping date, stopping date index) per path.
    """
    K, m = model["K"], model["m"]
    n = S1.shape[0]
    active = np.ones(n, dtype=bool)
    payoff = np.zeros(n)
    ex_idx = np.full(n, m, dtype=int)

    for i in range(first, m):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        h = np.maximum(np.maximum(S1[idx, i], S2[idx, i]) - K, 0.0)
        itm = h > 0                       # exercise requires h > C_hat >= 0
        cand = idx[itm]
        if cand.size == 0:
            continue
        C_hat = continuation_value(model, i, S1[cand, i], S2[cand, i])
        ex = h[itm] > C_hat
        stop = cand[ex]
        payoff[stop] = h[itm][ex]
        ex_idx[stop] = i
        active[stop] = False

    # Terminal payoff for paths that reach maturity
    rest = np.flatnonzero(active)
    payoff[rest] = np.maximum(np.maximum(S1[rest, m], S2[rest, m]) - K, 0.0)
    return payoff, ex_idx


def discounted_policy_payoffs(model: dict, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Pathwise payoff of the trained policy, discounted to t = 0."""
    payoff, ex_idx = apply_policy(model, S1, S2)
    dt = model["T"] / model["m"]
    return payoff * np.exp(-model["r"] * dt * ex_idx)


def lsm_price_max(model: dict, price1: np.ndarray, price2: np.ndarray,
                  return_se: bool = False) -> float | tuple[float, float]:
    """
    Forward pass (out-of-sample pricing). Identical for all methods: the decision rule
    depends only on the trained coefficients (exercise at date i if
    h_i > max(phi(S_i)' beta_i, 0), otherwise continue; payoff at maturity if never exercised).

    The exercise policy is learned on separate training paths and applied to the independent
    pricing paths given here, so the estimator is low-biased: E[price] <= true Bermudan price.

    If return_se is True, also returns the standard error of the estimate: sample std
    (ddof=1) of the discounted pathwise payoffs divided by sqrt(n). It reflects the noise of
    the pricing paths only, conditional on the trained policy (training noise not included).
    """
    discounted = discounted_policy_payoffs(model, price1, price2)
    price = float(np.mean(discounted))
    if return_se:
        se = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        return price, se
    return price


def european_discounted_payoff(S1: np.ndarray, S2: np.ndarray, K: float, r: float,
                               T: float) -> np.ndarray:
    """Discounted European max-call payoff e^{-rT} max(max(S1_T, S2_T) - K, 0) per path."""
    return np.exp(-r * T) * np.maximum(np.maximum(S1[:, -1], S2[:, -1]) - K, 0.0)


def lsm_price_max_cv(model: dict, train1: np.ndarray, train2: np.ndarray,
                     price1: np.ndarray, price2: np.ndarray, euro_price: float) -> dict:
    """
    Forward pass with the discounted European max-call payoff X as control variate:

        V_cv = mean(Y - b (X - E[X])),   E[X] = euro_price (Stulz closed form),

    where Y is the discounted payoff of the trained policy on the pricing paths. The
    coefficient b = Cov(Y, X) / Var(X) is estimated on the *training* paths (policy applied
    in sample), so it is independent of the pricing paths and V_cv is an exactly unbiased
    estimator of the policy value; hence it stays low-biased for the Bermudan price.

    Returns price and SE with and without control variate, b, and the variance reduction
    factor Var(Y) / Var(Y - b X) on the pricing paths.
    """
    K, r, T = model["K"], model["r"], model["T"]
    y_tr = discounted_policy_payoffs(model, train1, train2)
    x_tr = european_discounted_payoff(train1, train2, K, r, T)
    cxy = np.cov(y_tr, x_tr, ddof=1)
    b = float(cxy[0, 1] / cxy[1, 1]) if cxy[1, 1] > 0 else 0.0

    y = discounted_policy_payoffs(model, price1, price2)
    x = european_discounted_payoff(price1, price2, K, r, T)
    z = y - b * (x - euro_price)
    n = len(y)
    var_y, var_z = float(np.var(y, ddof=1)), float(np.var(z, ddof=1))
    return dict(price=float(z.mean()), se=float(np.sqrt(var_z / n)),
                price_plain=float(y.mean()), se_plain=float(np.sqrt(var_y / n)),
                b=b, vrf=var_y / var_z if var_z > 0 else np.inf,
                corr=float(np.corrcoef(y, x)[0, 1]))
