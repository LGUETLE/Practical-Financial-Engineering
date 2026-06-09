import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats  # <-- NEW: Required for Confidence Intervals

from american_options_functions import (
    simulate_gbm, 
    basis_max, 
    lsm_train_max, 
    lsm_price_max
)

# ==============================================================================
# ANALYSIS & PLOTTING FUNCTIONS
# ==============================================================================

def analyze_moneyness_sensitivity(T: float, r: float, delta: float, sigma: float, 
                                  K: float, m_steps: int, n_paths: int):
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
        # Generate training paths
        train_S1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, seed=100+int(S0))
        train_S2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, seed=200+int(S0))
        
        # Generate pricing paths
        price_S1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, seed=300+int(S0))
        price_S2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths, seed=400+int(S0))
        
        # Train and price (Explicitly using TVR method)
        model = lsm_train_max(train_S1, train_S2, K, r, T, method="TVR")
        price_lsm = lsm_price_max(model, price_S1, price_S2)
        
        exact = exact_prices[S0]
        abs_error = abs(price_lsm - exact)
        
        results.append({
            "S0": S0,
            "LSM_Price": round(price_lsm, 3),
            "Exact_Price": exact,
            "Abs_Error": round(abs_error, 3)
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    return df_results

def analyze_convergence(S0: float, K: float, T: float, r: float, delta: float, sigma: float, 
                        m_steps: int, path_counts: list):
    """
    Evaluates empirical convergence and computational time as N grows.
    """
    print("\n--- Running Convergence Analysis ---")
    
    results = []
    
    for n in path_counts:
        start_time = time.time()
        
        # Simulation
        t1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, seed=42)
        t2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, seed=43)
        p1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, seed=44)
        p2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n, seed=45)
        
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
    Phi = basis_max(S1_flat, S2_flat, K)
    C_hat_flat = np.maximum(Phi @ beta, 0.0)
    
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
    plt.close()

def analyze_volatility_sensitivity(S0: float, K: float, T: float, r: float, 
                                   delta: float, m_steps: int, n_paths: int):
    """
    Evaluates the sensitivity of the Max-Option price to changes in volatility.
    Tests sigma values: 10%, 20% (Baseline), and 30%.
    """
    print("\n--- Running Volatility Sensitivity Analysis ---")
    
    # Test different volatility levels
    sigma_values = [0.10, 0.20, 0.30]
    results = []
    
    for sig in sigma_values:
        # Generate training paths with the current sigma
        train_S1 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, seed=500+int(sig*100))
        train_S2 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, seed=600+int(sig*100))
        
        # Generate pricing paths with the current sigma
        price_S1 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, seed=700+int(sig*100))
        price_S2 = simulate_gbm(S0, sig, r, delta, T, m_steps, n_paths, seed=800+int(sig*100))
        
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
                           m_steps: int, n_paths: int, num_runs: int = 50):
    """
    Empirically compares the variance of Classic LSM and TVR 
    under theoretical laboratory conditions across multiple independent runs.
    Now includes 95% Confidence Intervals!
    """
    print(f"\n--- Running Theory Comparison: Classic LSM vs TVR ({num_runs} Runs) ---")
    
    lsm_prices = []
    tvr_prices = []
    
    for i in range(num_runs):
        # Generate new independent paths for each run
        t1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
        t2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
        p1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
        p2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
        
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
    
    # Export only the summary table to LaTeX
    df_summary.to_latex("output_tables/table_theory_comparison.tex", index=False, float_format="%.4f")
    
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
    t1_plot = simulate_gbm(S0_base, sigma_val, r_val, delta_val, T_val, m_steps, 20000, seed=1)
    t2_plot = simulate_gbm(S0_base, sigma_val, r_val, delta_val, T_val, m_steps, 20000, seed=2)
    
    # Train the model specifically for the plot (Explicitly using TVR method)
    plot_model = lsm_train_max(t1_plot, t2_plot, K_val, r_val, T_val, method="TVR")
    
    # Call the plotting function 
    plot_exercise_boundary(plot_model, step_index=4)
    print("Plot successfully saved to '/output_plots'.")
    
    print("\nExecution finished successfully.")