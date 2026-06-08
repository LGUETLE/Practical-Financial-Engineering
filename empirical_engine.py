import os
import numpy as np
import pandas as pd
import scipy.stats as stats

from american_options_functions import (
    simulate_correlated_gbm, 
    lsm_train_max, 
    lsm_price_max
)
# --- NEW: Import the Data Fetcher ---
from data_fetcher import HistoricalDataFetcher

# ==============================================================================
# EMPIRICAL ANALYSIS & STATISTICS
# ==============================================================================

def run_empirical_macro_replications(S0_1: float, S0_2: float, 
                                     mu_1: float, mu_2: float, 
                                     sigma_1: float, sigma_2: float, rho: float, 
                                     K: float, r: float, T: float, m_steps: int, 
                                     n_paths: int = 10000, R_runs: int = 50):
    """
    Executes macro-replications on correlated, historical market data and 
    calculates 95% confidence intervals to compare Classic LSM and TVR.
    """
    print(f"\n--- Running {R_runs} Macro-Replications (Real Market Data) ---")
    
    results_lsm = []
    results_tvr = []
    
    # IMPORTANT: For option pricing under the risk-neutral measure (Q), we must use 
    # the risk-free rate (r) as the drift parameter, NOT the historical mu! 
    # The historical mu is only used if we simulate real-world (P) scenarios.
    drift_1_rn = r
    drift_2_rn = r

    for run in range(R_runs):
        # Simulate correlated training and pricing paths for the current run
        t1, t2 = simulate_correlated_gbm(S0_1, S0_2, drift_1_rn, drift_2_rn, 
                                         sigma_1, sigma_2, rho, T, m_steps, n_paths)
        p1, p2 = simulate_correlated_gbm(S0_1, S0_2, drift_1_rn, drift_2_rn, 
                                         sigma_1, sigma_2, rho, T, m_steps, n_paths)
        
        # Pricing via Classic LSM (Tracking cashflows & ITM filter)
        model_lsm = lsm_train_max(t1, t2, K, r, T, method="LSM")
        results_lsm.append(lsm_price_max(model_lsm, p1, p2))
        
        # Pricing via TVR Value Iteration (Smoothed target variable)
        model_tvr = lsm_train_max(t1, t2, K, r, T, method="TVR")
        results_tvr.append(lsm_price_max(model_tvr, p1, p2))
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{R_runs} independent runs...")

    # --- Statistical Evaluation (Confidence Intervals) ---
    def compute_ci(prices):
        prices = np.array(prices)
        mean_val = np.mean(prices)
        var_val = np.var(prices, ddof=1)
        se = stats.sem(prices) # Standard Error of the Mean
        
        # Calculate 95% Confidence Interval using the t-distribution
        ci_low, ci_high = stats.t.interval(0.95, len(prices)-1, loc=mean_val, scale=se)
        return mean_val, var_val, ci_low, ci_high

    lsm_mean, lsm_var, lsm_low, lsm_high = compute_ci(results_lsm)
    tvr_mean, tvr_var, tvr_low, tvr_high = compute_ci(results_tvr)
    
    # Compile results into a DataFrame
    df_stats = pd.DataFrame({
        "Method": ["Classic LSM", "TVR Value Iteration"],
        "Mean_Price": [lsm_mean, tvr_mean],
        "Variance": [lsm_var, tvr_var],
        "CI_95_Lower": [lsm_low, tvr_low],
        "CI_95_Upper": [lsm_high, tvr_high]
    })
    
    print("\nStatistical Results (Calibrated to Real Data):")
    print(df_stats.to_string(index=False))
    
    # Export to LaTeX for the final project documentation
    os.makedirs("output_tables", exist_ok=True)
    df_stats.to_latex("output_tables/table_empirical_macro.tex", index=False, float_format="%.4f")
    
    return df_stats

# ==============================================================================
# MAIN EXECUTION (Updated to use Data Fetcher)
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Initialize the fetcher and get real market data
    # We use Apple (AAPL) and Microsoft (MSFT) as highly correlated tech stocks.
    # You can change these tickers or dates at any time!
    fetcher = HistoricalDataFetcher(
        ticker_1="AAPL", 
        ticker_2="MSFT", 
        start_date="2021-01-01" # Fetching roughly 5 years of history
    )
    
    price_df = fetcher.fetch_data()
    market_params = fetcher.calculate_historical_parameters(price_df)
    
    # 2. Define the Option Contract Specifications
    # We set the strike price (K) to roughly match the current asset prices 
    # to ensure the option is traded around the At-The-Money (ATM) level.
    
    K_strike = 150.0 
    r_riskfree = 0.05
    T_mat = 1.0         # 1 year maturity
    m_exercise = 12     # 12 early exercise opportunities (e.g., monthly)
    
    n_paths = 10000     # Paths per simulation
    R_runs = 50         # Number of macro-replications
    
    # 3. Execute the statistical engine using the dynamically fetched parameters
    run_empirical_macro_replications(
        S0_1=market_params["S0_1"], 
        S0_2=market_params["S0_2"], 
        mu_1=market_params["mu_1"], 
        mu_2=market_params["mu_2"], 
        sigma_1=market_params["sigma_1"], 
        sigma_2=market_params["sigma_2"], 
        rho=market_params["rho"], 
        K=K_strike, 
        r=r_riskfree, 
        T=T_mat, 
        m_steps=m_exercise, 
        n_paths=n_paths, 
        R_runs=R_runs
    )