import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt  # <-- NEW: Added for empirical plotting

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
                                     scenario_name: str = "tech"):  # <-- NEW: Added scenario name
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
        # Re-calculating high bound explicitly to avoid unpacking issues if needed
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
    
    # --- NEW: Generate and Export Boxplot ---
    os.makedirs("output_plots", exist_ok=True)
    plt.figure(figsize=(6, 4.5))
    
    # Plotting both datasets side by side
    plt.boxplot([results_lsm, results_tvr], labels=['Classic LSM', 'TVR Value Iteration'])
    plt.title(f"Empirical Price Distribution: {scenario_name.upper()} Scenario")
    plt.ylabel("Calculated Option Price ($V_0$)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"output_plots/boxplot_variance_{scenario_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to 'output_plots/boxplot_variance_{scenario_name}.pdf'")
    
    # --- Export to unique LaTeX table file ---
    os.makedirs("output_tables", exist_ok=True)
    df_stats.to_latex(f"output_tables/table_empirical_{scenario_name}.tex", index=False, float_format="%.4f")
    print(f"LaTeX table saved to 'output_tables/table_empirical_{scenario_name}.tex'")
    
    return df_stats

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
        mu_1=tech_params["mu_1"], mu_2=tech_params["mu_2"], 
        sigma_1=tech_params["sigma_1"], sigma_2=tech_params["sigma_2"], 
        rho=tech_params["rho"], 
        K=K_strike, r=r_riskfree, T=T_mat, m_steps=m_exercise,  # <-- Fixed: T=T_mat
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
        mu_1=crypto_params["mu_1"], mu_2=crypto_params["mu_2"], 
        sigma_1=crypto_params["sigma_1"], sigma_2=crypto_params["sigma_2"], 
        rho=crypto_params["rho"], 
        K=K_strike, r=r_riskfree, T=T_mat, m_steps=m_exercise,  # <-- Fixed: T=T_mat
        n_paths=n_paths, R_runs=R_runs, 
        scenario_name="crypto" 
    )