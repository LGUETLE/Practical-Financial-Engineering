import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from american_options_functions import (
    simulate_correlated_gbm, 
    lsm_train_max, 
    lsm_price_max
)
from data_fetcher import HistoricalDataFetcher

# ==============================================================================
# EMPIRICAL ANALYSIS & STATISTICS
# ==============================================================================

def run_empirical_macro_replications(S0_1: float, S0_2: float, 
                                     mu_1: float, mu_2: float, 
                                     sigma_1: float, sigma_2: float, rho: float, 
                                     K: float, r: float, T: float, m_steps: int, 
                                     n_paths: int = 10000, R_runs: int = 50,
                                     scenario_name: str = "tech"):
    """
    Executes macro-replications on correlated, historical market data and 
    calculates 95% confidence intervals to compare Classic LSM and TVR.
    Also generates a comparison boxplot for visual variance analysis.
    """
    print(f"\n--- Running {R_runs} Macro-Replications ({scenario_name.upper()} Data) ---")
    
    results_lsm = []
    results_tvr = []
    
    # Risk-neutral measure dynamics (r is used as the drift for pricing)
    drift_1_rn = r
    drift_2_rn = r

    for run in range(R_runs):
        t1, t2 = simulate_correlated_gbm(S0_1, S0_2, drift_1_rn, drift_2_rn, 
                                         sigma_1, sigma_2, rho, T, m_steps, n_paths)
        p1, p2 = simulate_correlated_gbm(S0_1, S0_2, drift_1_rn, drift_2_rn, 
                                         sigma_1, sigma_2, rho, T, m_steps, n_paths)
        
        # Classic LSM
        model_lsm = lsm_train_max(t1, t2, K, r, T, method="LSM")
        results_lsm.append(lsm_price_max(model_lsm, p1, p2))
        
        # TVR Value Iteration
        model_tvr = lsm_train_max(t1, t2, K, r, T, method="TVR")
        results_tvr.append(lsm_price_max(model_tvr, p1, p2))
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{R_runs} independent runs...")

    # --- Statistical Evaluation (Confidence Intervals) ---
    def compute_ci(prices):
        prices = np.array(prices)
        mean_val = np.mean(prices)
        var_val = np.var(prices, ddof=1)
        se = stats.sem(prices)
        ci_low, _ = stats.t.interval(0.95, len(prices)-1, loc=mean_val, scale=se)
        _, ci_high = stats.t.interval(0.95, len(prices)-1, loc=mean_val, scale=se)
        return mean_val, var_val, ci_low, ci_high

    lsm_mean, lsm_var, lsm_low, lsm_high = compute_ci(results_lsm)
    tvr_mean, tvr_var, tvr_low, tvr_high = compute_ci(results_tvr)
    
    df_stats = pd.DataFrame({
        "Method": ["Classic LSM", "TVR Value Iteration"],
        "Mean_Price": [lsm_mean, tvr_mean],
        "Variance": [lsm_var, tvr_var],
        "CI_95_Lower": [lsm_low, tvr_low],
        "CI_95_Upper": [lsm_high, tvr_high]
    })
    
    print(f"\nStatistical Results ({scenario_name.upper()}):")
    print(df_stats.to_string(index=False))
    
    # --- Generate and Export Boxplot ---
    os.makedirs("output_plots", exist_ok=True)
    plt.figure(figsize=(6, 4.5))
    plt.boxplot([results_lsm, results_tvr], labels=['Classic LSM', 'TVR Value Iteration'])
    plt.title(f"Empirical Price Distribution: {scenario_name.upper()} Scenario")
    plt.ylabel("Calculated Option Price ($V_0$)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output_plots/boxplot_variance_{scenario_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
    # --- Export to unique LaTeX table file ---
    os.makedirs("output_tables", exist_ok=True)
    df_stats.to_latex(f"output_tables/table_empirical_{scenario_name}.tex", index=False, float_format="%.4f")
    
    return df_stats

# ==============================================================================
# SENSITIVITY PLOT (2D MAX OPTION)
# ==============================================================================

def plot_2d_sigma_sensitivity(K, r, T, m_steps, n_paths, tech_vola, crypto_vola, sigmas=None):
    """
    2D Max-Option Sensitivity Analysis: Classic LSM vs TVR.
    Dynamically visualizes the empirical volatility regimes (Tech vs. Crypto) fetched earlier.
    """
    print("\n" + "="*60)
    print("STARTING 2D SENSITIVITY ANALYSIS (LSM vs TVR)")
    print("="*60)
    
    if sigmas is None:
        sigmas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

    lsm_prices, tvr_prices = [], []
    
    # Constant correlation to isolate the volatility effect in the plot
    test_rho = 0.5 
    S0_norm = 100.0

    for sig in sigmas:
        # Generate training and pricing paths using your correlated GBM function
        t1, t2 = simulate_correlated_gbm(S0_norm, S0_norm, r, r, sig, sig, test_rho, T, m_steps, n_paths)
        p1, p2 = simulate_correlated_gbm(S0_norm, S0_norm, r, r, sig, sig, test_rho, T, m_steps, n_paths)
        
        # LSM
        model_lsm = lsm_train_max(t1, t2, K, r, T, method="LSM")
        p_lsm = lsm_price_max(model_lsm, p1, p2)
        
        # TVR
        model_tvr = lsm_train_max(t1, t2, K, r, T, method="TVR")
        p_tvr = lsm_price_max(model_tvr, p1, p2)
        
        lsm_prices.append(p_lsm)
        tvr_prices.append(p_tvr)
        print(f"  σ={sig:.2f} | LSM={p_lsm:.4f} | TVR={p_tvr:.4f}")

    # --- Plotting ---
    os.makedirs("output_plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)

    # Plot 1: Options Price vs. Volatility
    ax = axes[0]
    ax.plot(sigmas, tvr_prices, "o-", color="crimson", lw=2, label="TVR Value Iteration")
    ax.plot(sigmas, lsm_prices, "s--", color="steelblue", lw=2, label="Classic LSM")
    
    # Dynamic Empirical Basket Markers
    ax.axvline(tech_vola, color="darkgreen", ls="-.", lw=1.5, label=f"Tech Regime (≈ {tech_vola:.2f})")
    ax.axvline(crypto_vola, color="darkorange", ls="-.", lw=1.5, label=f"Crypto Regime (≈ {crypto_vola:.2f})")
    
    ax.set(xlabel="Volatility (σ)", ylabel="Option Price ($V_0$)", title="2D Max-Option: Price vs. Volatility")
    ax.legend(fontsize=8)

    # Plot 2: Absolute Difference (TVR - LSM)
    ax = axes[1]
    diffs = [abs(t - l) for t, l in zip(tvr_prices, lsm_prices)]
    ax.bar([f"{s:.2f}" for s in sigmas], diffs, color="steelblue", alpha=0.8)
    ax.set(xlabel="Volatility (σ)", ylabel="|TVR − LSM|", title="Pricing Divergence at High Volatility")

    plt.tight_layout()
    plt.savefig("output_plots/05_empirical_2d_sensitivity.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Sensitivity plot saved to 'output_plots/05_empirical_2d_sensitivity.pdf'")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # Contract specifications
    K_strike = 100.0    
    r_riskfree = 0.05   
    T_mat = 1.0         
    m_exercise = 12     
    n_paths = 10000     
    R_runs = 50         
    S0_norm = 100.0 
    
    # ---------------------------------------------------------
    # SCENARIO 1: TRADITIONAL TECH STOCKS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING SCENARIO 1: TECH STOCKS (AAPL vs MSFT)")
    print("="*60)
    
    tech_fetcher = HistoricalDataFetcher(
        ticker_1="AAPL", ticker_2="MSFT", start_date="2022-01-01", asset_class="stock"
    )
    tech_df = tech_fetcher.fetch_data()
    tech_params = tech_fetcher.calculate_historical_parameters(tech_df)
    
    run_empirical_macro_replications(
        S0_1=S0_norm, S0_2=S0_norm, 
        mu_1=r_riskfree, mu_2=r_riskfree, 
        sigma_1=tech_params["sigma_1"], sigma_2=tech_params["sigma_2"], 
        rho=tech_params["rho"], 
        K=K_strike, r=r_riskfree, T=T_mat, m_steps=m_exercise,
        n_paths=n_paths, R_runs=R_runs, 
        scenario_name="tech" 
    )
    
    # ---------------------------------------------------------
    # SCENARIO 2: CRYPTOCURRENCY
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("STARTING SCENARIO 2: CRYPTOCURRENCY (BTC vs ETH)")
    print("="*60)
    
    crypto_fetcher = HistoricalDataFetcher(
        ticker_1="BTC", ticker_2="ETH", start_date="2022-01-01", asset_class="crypto"
    )
    crypto_df = crypto_fetcher.fetch_data()
    crypto_params = crypto_fetcher.calculate_historical_parameters(crypto_df)
    
    run_empirical_macro_replications(
        S0_1=S0_norm, S0_2=S0_norm, 
        mu_1=r_riskfree, mu_2=r_riskfree, 
        sigma_1=crypto_params["sigma_1"], sigma_2=crypto_params["sigma_2"], 
        rho=crypto_params["rho"], 
        K=K_strike, r=r_riskfree, T=T_mat, m_steps=m_exercise,
        n_paths=n_paths, R_runs=R_runs, 
        scenario_name="crypto" 
    )

    # ---------------------------------------------------------
    # SCENARIO 3: 2D SENSITIVITY PLOT (TECH VS CRYPTO)
    # ---------------------------------------------------------
    # Calculate average volatility for each basket to place the vertical lines
    avg_tech_vola = (tech_params["sigma_1"] + tech_params["sigma_2"]) / 2
    avg_crypto_vola = (crypto_params["sigma_1"] + crypto_params["sigma_2"]) / 2

    plot_2d_sigma_sensitivity(
        K=K_strike, 
        r=r_riskfree, 
        T=T_mat, 
        m_steps=m_exercise, 
        n_paths=n_paths, 
        tech_vola=avg_tech_vola, 
        crypto_vola=avg_crypto_vola
    )