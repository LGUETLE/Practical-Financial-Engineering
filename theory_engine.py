import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats  # <-- NEW: Required for Confidence Intervals

from american_options_functions import (
    simulate_gbm, 
    basis_max, 
    continuation_value,
    lsm_train_max, 
    lsm_price_max,
    DEFAULT_DEGREE,
    DEFAULT_SCALE,
)
from mc_stats import summarize, paired_t, holm, df_to_markdown, write_result

# Glasserman (2003), Example 8.6.1: two independent assets, 9 exercise dates.
BENCH = dict(K=100.0, T=3.0, r=0.05, delta=0.10, sigma=0.20, m=9)
# Reference prices from Glasserman (binomial lattice).
REFERENCE = {90.0: 8.08, 100.0: 13.90, 110.0: 21.34}


def benchmark_paths(S0: float, n_train: int, n_price: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Training and independent pricing paths (two independent GBMs each) for the benchmark."""
    b = BENCH
    t1 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], n_train, rng)
    t2 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], n_train, rng)
    p1 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], n_price, rng)
    p2 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], n_price, rng)
    return t1, t2, p1, p2


def run_basis_ablation(S0: float = 100.0, R: int = 50, n_paths: int = 4000,
                       seed: int = 2026) -> tuple[pd.DataFrame, dict]:
    """
    Basis ablation on the Glasserman benchmark: basis {degree 2, degree 3} x scaling
    {raw S, S/K} for classic LSM and TVR. Common random numbers: in replication k all eight
    configurations are trained and priced on the same training and pricing paths
    (generator default_rng([seed, 1, k])). Reports mean, SE, bias, RMSE against the reference
    and the 2-norm condition numbers of the regression matrices (over all dates and runs).
    All prices are low-biased, so the basis with the higher mean price gives the better policy.
    """
    print(f"\n--- Basis ablation (S0 = {S0}, R = {R}, N = {n_paths}) ---")
    b = BENCH
    configs = [(meth, deg, sc) for meth in ("LSM", "TVR") for deg in (2, 3) for sc in (False, True)]
    prices = {c: np.empty(R) for c in configs}
    conds = {c: [] for c in configs}
    for k in range(R):
        rng = np.random.default_rng([seed, 1, k])
        t1, t2, p1, p2 = benchmark_paths(S0, n_paths, n_paths, rng)
        for c in configs:
            model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=c[0],
                                  degree=c[1], scale=c[2], diagnostics=True)
            prices[c][k] = lsm_price_max(model, p1, p2)
            conds[c].extend(model["cond"].values())

    ref = REFERENCE[S0]
    rows = []
    for c in configs:
        st = summarize(prices[c], ref)
        lc = np.log10(conds[c])
        rows.append({"Method": "Classic LSM" if c[0] == "LSM" else "TVR",
                     "Basis": f"deg {c[1]} ({7 if c[1] == 2 else 11} terms)",
                     "Scaling": "S/K" if c[2] else "raw",
                     "Mean": st["mean"], "SE": st["se"], "Bias": st["bias"], "RMSE": st["rmse"],
                     "log10 cond (median)": float(np.median(lc)),
                     "log10 cond (max)": float(np.max(lc))})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Scaling: identical policy up to floating point?
    scale_rows = []
    for meth in ("LSM", "TVR"):
        for deg in (2, 3):
            d = prices[(meth, deg, True)] - prices[(meth, deg, False)]
            scale_rows.append({"Method": meth, "Degree": deg,
                               "max abs. difference over runs": float(np.max(np.abs(d))),
                               "runs with any difference": int(np.sum(d != 0.0))})
    df_scale = pd.DataFrame(scale_rows)
    print(df_scale.to_string(index=False))

    # Degree: paired t-test deg 3 - deg 2 (scaled), Holm over the two methods
    tests = [paired_t(prices[(meth, 3, True)], prices[(meth, 2, True)]) for meth in ("LSM", "TVR")]
    p_holm = holm([t["p"] for t in tests])
    deg_rows = []
    for meth, t, ph in zip(("Classic LSM", "TVR"), tests, p_holm):
        deg_rows.append({"Method": meth, "Mean diff deg3 - deg2": t["diff"], "SE": t["se"],
                         "t": t["t"], "p": t["p"], "p (Holm)": ph,
                         "95% CI": f"[{t['ci_low']:.4f}, {t['ci_high']:.4f}]"})
    df_deg = pd.DataFrame(deg_rows)
    print(df_deg.to_string(index=False))

    # Choice of default basis: higher mean price, averaged over both methods (scaled)
    mean_deg = {deg: np.mean([np.mean(prices[(meth, deg, True)]) for meth in ("LSM", "TVR")])
                for deg in (2, 3)}
    chosen = max(mean_deg, key=mean_deg.get)
    choice = dict(degree=chosen, scale=True, mean_by_degree=mean_deg)
    print(f"Mean price by degree (avg over LSM, TVR; scaled): {mean_deg} -> default degree {chosen}")

    text = (f"# Basis ablation (Glasserman benchmark, S0 = {S0:g})\n\n"
            f"R = {R} replications, N = {n_paths:,} training and {n_paths:,} independent pricing "
            f"paths per replication; all eight configurations use the same paths in each "
            f"replication (common random numbers, seed {seed}). Reference {ref:.2f}. SE = standard "
            f"error of the mean over replications. Condition numbers (2-norm) of the regression "
            f"matrix over all exercise dates and replications (LSM: ITM rows only). "
            f"Generated by `theory_engine.py`.\n\n"
            + df_to_markdown(df, {"Mean": "{:.4f}", "SE": "{:.4f}", "Bias": "{:+.4f}",
                                  "RMSE": "{:.4f}", "log10 cond (median)": "{:.1f}",
                                  "log10 cond (max)": "{:.1f}"})
            + "\n\nScaled vs raw basis on the same paths (difference of the price estimates):\n\n"
            + df_to_markdown(df_scale, {"max abs. difference over runs": "{:.2e}"})
            + "\n\nPaired t-test, degree 3 vs degree 2 (scaled), Holm correction over the two "
              "methods:\n\n"
            + df_to_markdown(df_deg, {"Mean diff deg3 - deg2": "{:+.4f}", "SE": "{:.4f}",
                                      "t": "{:.3f}", "p": "{:.4f}", "p (Holm)": "{:.4f}"})
            + f"\n\nMean price by degree (average of LSM and TVR, scaled basis): "
              f"degree 2: {mean_deg[2]:.4f}, degree 3: {mean_deg[3]:.4f}.\n")
    write_result("basis_ablation", text)
    os.makedirs("output_tables", exist_ok=True)
    df.to_csv("output_tables/basis_ablation.csv", index=False)
    return df, choice


def run_benchmark_replications(methods: tuple[str, ...] = ("LSM", "TVR"),
                               S0_values: tuple[float, ...] = (90.0, 100.0, 110.0),
                               R: int = 50, n_paths: int = 4000, seed: int = 2026,
                               degree: int = DEFAULT_DEGREE,
                               scale: bool = DEFAULT_SCALE) -> dict:
    """
    R independent replications of the Glasserman benchmark for each S0. In replication k all
    methods use the same training and pricing paths (common random numbers), and the same
    normals are reused across S0 (generator default_rng([seed, 2, k])).
    Returns {(S0, method): array of R low-biased price estimates}.
    """
    b = BENCH
    out = {(S0, meth): np.empty(R) for S0 in S0_values for meth in methods}
    for S0 in S0_values:
        for k in range(R):
            rng = np.random.default_rng([seed, 2, k])
            t1, t2, p1, p2 = benchmark_paths(S0, n_paths, n_paths, rng)
            for meth in methods:
                model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=meth,
                                      degree=degree, scale=scale)
                out[(S0, meth)][k] = lsm_price_max(model, p1, p2)
    return out


def summarize_benchmark(prices: dict, methods: tuple[str, ...]) -> pd.DataFrame:
    """Mean, SE, 95% CI, bias and RMSE against the Glasserman reference per (S0, method)."""
    rows = []
    for (S0, meth), x in prices.items():
        if meth not in methods:
            continue
        st = summarize(x, REFERENCE[S0])
        rows.append({"S0": S0, "Method": meth, "Mean": st["mean"], "SE": st["se"],
                     "CI low": st["ci_low"], "CI high": st["ci_high"],
                     "Reference": REFERENCE[S0], "Bias": st["bias"], "RMSE": st["rmse"]})
    return pd.DataFrame(rows)

# ==============================================================================
# ANALYSIS & PLOTTING FUNCTIONS
# ==============================================================================

def analyze_moneyness_sensitivity(T: float, r: float, delta: float, sigma: float, 
                                  K: float, m_steps: int, n_paths: int,
                                  seed: int = 2026) -> pd.DataFrame:
    """
    Evaluates the algorithm across different initial prices (Moneyness).
    Replicates the structure of Glasserman Table 8.1 and 8.2.
    """
    print("\n--- Running Moneyness Sensitivity Analysis ---")
    
    # S0 values: Out-of-the-money (90), At-the-money (100), In-the-money (110)
    S0_values = [90.0, 100.0, 110.0]
    
    # Exact prices from Glasserman (calculated via Binomial Lattice)
    exact_prices = {90.0: 8.08, 100.0: 13.90, 110.0: 21.34}
    
    results = []
    
    for S0 in S0_values:
        # One generator per S0 (seeded by (seed, S0)), passed explicitly
        rng = np.random.default_rng([seed, int(S0)])

        # Generate training paths
        train_S1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        train_S2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        
        # Generate pricing paths
        price_S1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        price_S2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        
        # Train and price (Explicitly using TVR method)
        model = lsm_train_max(train_S1, train_S2, K, r, T, method="TVR")
        price_lsm, se_lsm = lsm_price_max(model, price_S1, price_S2, return_se=True)

        exact = exact_prices[S0]
        diff = price_lsm - exact

        results.append({
            "S0": S0,
            "LSM_Price": round(price_lsm, 3),
            "Std_Error": round(se_lsm, 3),
            "Exact_Price": exact,
            "Difference": round(diff, 3),
            "Diff_over_SE": round(diff / se_lsm, 2)
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    return df_results

def analyze_convergence(S0: float, K: float, T: float, r: float, delta: float, sigma: float, 
                        m_steps: int, path_counts: list,
                        seed: int = 2026) -> pd.DataFrame:
    """
    Evaluates empirical convergence and computational time as N grows.
    """
    print("\n--- Running Convergence Analysis ---")
    
    results = []
    
    for n in path_counts:
        start_time = time.time()
        rng = np.random.default_rng([seed, n])
        
        # Simulation
        t1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, rng)
        t2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, rng)
        p1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, rng)
        p2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, rng)
        
        # Algorithm (Explicitly using TVR method)
        model = lsm_train_max(t1, t2, K, r, T, method="TVR")
        price = lsm_price_max(model, p1, p2)
        
        elapsed_time = time.time() - start_time
        
        results.append({
            "Paths (N)": n,
            "LSM_Price": round(price, 3),
            "Time (sec)": round(elapsed_time, 3)
        })
        print(f"Computed N={n} in {elapsed_time:.2f}s -> Price: {price:.3f}")
        
    df_results = pd.DataFrame(results)
    return df_results

def plot_exercise_boundary(model: dict, step_index: int = 4):
    """
    Plots the estimated exercise region for a specific time step.
    Replicates Glasserman Figure 8.9.
    """
    betas = model["betas"]
    if step_index not in betas:
        print(f"Step {step_index} not found in model.")
        return
        
    K = model["K"]
    beta = betas[step_index]
    
    # Create a grid of S1 and S2 values
    s_values = np.linspace(60, 140, 200)
    S1_grid, S2_grid = np.meshgrid(s_values, s_values)
    
    S1_flat = S1_grid.flatten()
    S2_flat = S2_grid.flatten()
    
    # Calculate intrinsic value on the grid
    h_flat = np.maximum(np.maximum(S1_flat, S2_flat) - K, 0.0)
    
    # Calculate continuation value on the grid
    C_hat_flat = continuation_value(model, step_index, S1_flat, S2_flat)
    
    # Determine where immediate exercise is optimal
    exercise_flat = h_flat > C_hat_flat
    exercise_grid = exercise_flat.reshape(S1_grid.shape)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot the exercise region (shaded)
    plt.contourf(S1_grid, S2_grid, exercise_grid, levels=[0.5, 1.5], 
                 colors=['lightgray'], alpha=0.8)
    
    # Formatting
    plt.axvline(x=K, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.axhline(y=K, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.title(f"Estimated Exercise Region at Step $i={step_index}$")
    plt.xlabel("$S_1$")
    plt.ylabel("$S_2$")
    plt.xlim(60, 140)
    plt.ylim(60, 140)
    plt.grid(alpha=0.3)
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightgray', edgecolor='gray', label='Exercise Region')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    # Save the figure as a high-quality vector graphic for LaTeX
    plt.savefig("output_plots/exercise_boundary.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("output_plots/exercise_boundary.png", format="png", dpi=150, bbox_inches="tight")
    plt.close()

def analyze_volatility_sensitivity(S0: float, K: float, T: float, r: float, 
                                   delta: float, m_steps: int, n_paths: int,
                                   seed: int = 2026) -> pd.DataFrame:
    """
    Evaluates the sensitivity of the Max-Option price to changes in volatility.
    Tests sigma values: 10%, 20% (Baseline), and 30%.
    """
    print("\n--- Running Volatility Sensitivity Analysis ---")
    
    # Test different volatility levels
    sigma_values = [0.10, 0.20, 0.30]
    results = []
    
    for sig in sigma_values:
        rng = np.random.default_rng([seed, int(round(sig * 100))])

        # Generate training paths with the current sigma
        train_S1 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, rng)
        train_S2 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, rng)
        
        # Generate pricing paths with the current sigma
        price_S1 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, rng)
        price_S2 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, rng)
        
        # Train and price (Explicitly using TVR method)
        model = lsm_train_max(train_S1, train_S2, K, r, T, method="TVR")
        price_lsm = lsm_price_max(model, price_S1, price_S2)
        
        results.append({
            "Volatility (sigma)": f"{sig*100:.0f}%",
            "LSM_Price": round(price_lsm, 3)
        })
        print(f"Computed for sigma={sig:.2f} -> Price: {price_lsm:.3f}")
        
    df_results = pd.DataFrame(results)
    return df_results

def compare_lsm_tvr_theory(S0: float, K: float, T: float, r: float, delta: float, sigma: float, 
                           m_steps: int, n_paths: int, num_runs: int = 50,
                           seed: int = 2026):
    """
    Empirically compares the variance of Classic LSM and TVR
    under theoretical laboratory conditions across multiple independent runs.
    Now includes 95% Confidence Intervals!

    A single generator np.random.default_rng(seed) is created here and passed
    explicitly to every simulation call, so the whole sequence of runs is
    reproducible without touching global NumPy state.
    """
    print(f"\n--- Running Theory Comparison: Classic LSM vs TVR ({num_runs} Runs) ---")
    rng = np.random.default_rng(seed)

    lsm_prices = []
    tvr_prices = []
    
    for i in range(num_runs):
        # Generate new independent paths for each run
        t1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        t2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        p1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        p2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, rng)
        
        # Evaluate Classic LSM
        model_lsm = lsm_train_max(t1, t2, K, r, T, method="LSM")
        lsm_prices.append(lsm_price_max(model_lsm, p1, p2))
        
        # Evaluate TVR Value Iteration
        model_tvr = lsm_train_max(t1, t2, K, r, T, method="TVR")
        tvr_prices.append(lsm_price_max(model_tvr, p1, p2))
        
        # Print progress to keep track during longer runs
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_runs} independent runs...")
            
    # --- Print Run-by-Run Table (Optional visual check) ---
    df_runs = pd.DataFrame({
        "Run": range(1, num_runs + 1),
        "Classic_LSM": lsm_prices,
        "TVR_Value_Iteration": tvr_prices
    })
    # Uncomment the next line if you want to see all 50 runs in the console
    # print(df_runs.to_string(index=False))

    # --- Statistical Evaluation (Confidence Intervals) ---
    def compute_stats(prices):
        prices = np.array(prices)
        mean_val = np.mean(prices)
        var_val = np.var(prices, ddof=1)
        se = stats.sem(prices) # Standard Error of the Mean
        
        # Calculate 95% Confidence Interval using the t-distribution
        ci_low, ci_high = stats.t.interval(0.95, len(prices)-1, loc=mean_val, scale=se)
        return mean_val, var_val, ci_low, ci_high

    lsm_mean, lsm_var, lsm_low, lsm_high = compute_stats(lsm_prices)
    tvr_mean, tvr_var, tvr_low, tvr_high = compute_stats(tvr_prices)
    
    # Compile summary results into a DataFrame for export
    df_summary = pd.DataFrame({
        "Method": ["Classic LSM", "TVR Value Iteration"],
        "Mean_Price": [lsm_mean, tvr_mean],
        "Variance": [lsm_var, tvr_var],
        "CI_95_Lower": [lsm_low, tvr_low],
        "CI_95_Upper": [lsm_high, tvr_high]
    })
    
    print("\nTheoretical Statistics (Glasserman Benchmark):")
    print(df_summary.to_string(index=False))

    # --- Test for equal variances ---
    # Two-sided F-test. It assumes independent samples; here LSM and TVR are
    # priced on the same paths in each run, so the Pitman-Morgan test for
    # paired samples (correlation of X+Y with X-Y) is reported as well.
    x, y = np.array(lsm_prices), np.array(tvr_prices)
    f_stat = lsm_var / tvr_var
    df1 = df2 = num_runs - 1
    f_p = 2 * min(stats.f.cdf(f_stat, df1, df2), stats.f.sf(f_stat, df1, df2))
    pm_r, pm_p = stats.pearsonr(x + y, x - y)
    pair_corr = np.corrcoef(x, y)[0, 1]
    print(f"\nF-test (H0: equal variances): F = {f_stat:.4f}, "
          f"df = ({df1}, {df2}), two-sided p = {f_p:.4f}")
    print(f"Correlation of LSM and TVR prices across runs: {pair_corr:.4f}")
    print(f"Pitman-Morgan test (paired): r = {pm_r:.4f}, p = {pm_p:.4f}")

    # --- Paired t-test for the mean difference TVR - LSM ---
    # Both prices are low-biased estimates on the same pricing paths, so a
    # positive mean difference indicates a better exercise policy for TVR.
    d = y - x
    d_mean = float(np.mean(d))
    d_se = float(stats.sem(d))
    t_res = stats.ttest_rel(y, x)
    d_low, d_high = stats.t.interval(0.95, num_runs - 1, loc=d_mean, scale=d_se)
    print(f"\nPaired t-test (H0: E[TVR - LSM] = 0): mean diff = {d_mean:.4f}, "
          f"SE = {d_se:.4f}, t = {t_res.statistic:.4f}, df = {num_runs - 1}, "
          f"two-sided p = {t_res.pvalue:.4g}, 95% CI = [{d_low:.4f}, {d_high:.4f}]")

    # Export only the summary table to LaTeX
    df_summary.to_latex("output_tables/table_theory_comparison.tex", index=False, float_format="%.4f")
    df_runs.to_csv("output_tables/theory_comparison_runs.csv", index=False)
    df_summary.to_csv("output_tables/theory_comparison_summary.csv", index=False)

    return df_summary

# ==============================================================================
# MAIN EXECUTION & EXPORT
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Setup export directories for LaTeX and PDF files
    print("Setting up export directories...")
    os.makedirs("output_tables", exist_ok=True)
    os.makedirs("output_plots", exist_ok=True)

    # 2. Define base parameters (matching Glasserman, Example 8.6.1)
    S0_base = 100.0     # Initial stock price
    K_val = 100.0       # Strike price
    T_val = 3.0         # Time to maturity in years
    r_val = 0.05        # Risk-free interest rate (5%)
    delta_val = 0.10    # Continuous dividend yield (10%)
    sigma_val = 0.20    # Volatility (20%)
    m_steps = 9         # Number of early exercise opportunities
    n_paths = 4000      # Number of paths for both training and pricing

    # 2.1 Basis ablation (degree x scaling, LSM and TVR)
    run_basis_ablation(S0=S0_base, R=50, n_paths=n_paths)

    # 2.2 Replicated benchmark at S0 = 90/100/110 with the default basis
    methods = ("LSM", "TVR")
    bench = run_benchmark_replications(methods, R=50, n_paths=n_paths)
    df_bench = summarize_benchmark(bench, methods)
    print(f"\nReplicated benchmark (R = 50, basis degree {DEFAULT_DEGREE}, scaled = {DEFAULT_SCALE}):")
    print(df_bench.to_string(index=False))

    # 3.0 Run theoretical comparison of Classic LSM vs TVR (Increased to 50 runs for statistical significance!)
    compare_lsm_tvr_theory(
        S0_base, K_val, T_val, r_val, delta_val, sigma_val, m_steps, n_paths, num_runs=50
    )

    # 3.1 Run sensitivity analysis (Moneyness) and export to LaTeX
    print("\nStarting Moneyness Sensitivity Analysis (Table 1)...")
    df_moneyness = analyze_moneyness_sensitivity(
        T_val, r_val, delta_val, sigma_val, K_val, m_steps, n_paths
    )
    
    # Export DataFrame directly to a LaTeX table format
    df_moneyness.to_latex(
        "output_tables/table_moneyness.tex", 
        index=False, 
        float_format="%.3f"
    )
    df_moneyness.to_csv("output_tables/table_moneyness.csv", index=False)

    # 3.2 Run Volatility Sensitivity and export
    print("\nStarting Volatility Sensitivity Analysis (Table 3)...")
    df_volatility = analyze_volatility_sensitivity(
        S0_base, K_val, T_val, r_val, delta_val, m_steps, n_paths
    )
    df_volatility.to_latex(
        "output_tables/table_volatility.tex", 
        index=False
    )

    # 4. Run convergence analysis (Path count) and export to LaTeX
    print("\nStarting Convergence Analysis (Table 2)...")
    path_grid = [1000, 4000, 10000, 40000]
    df_convergence = analyze_convergence(
        S0_base, K_val, T_val, r_val, delta_val, sigma_val, m_steps, path_grid
    )
    
    df_convergence.to_latex(
        "output_tables/table_convergence.tex", 
        index=False, 
        float_format="%.3f"
    )
    print("\nAll tables successfully exported to '/output_tables'.")

    # 5. Generate and export the exercise boundary plot (Contour Plot)
    print("\nGenerating Exercise Boundary Plot...")
    
    # Simulate a high-density path set to ensure a smooth exercise boundary
    rng_plot = np.random.default_rng(1)
    t1_plot = simulate_gbm(S0_base, sigma_val, r_val, delta_val, T_val, m_steps, 20000, rng_plot)
    t2_plot = simulate_gbm(S0_base, sigma_val, r_val, delta_val, T_val, m_steps, 20000, rng_plot)
    
    # Train the model specifically for the plot (Explicitly using TVR method)
    plot_model = lsm_train_max(t1_plot, t2_plot, K_val, r_val, T_val, method="TVR")
    
    # Call the plotting function 
    plot_exercise_boundary(plot_model, step_index=4)
    print("Plot successfully saved to '/output_plots'.")
    
    print("\nExecution finished successfully.")