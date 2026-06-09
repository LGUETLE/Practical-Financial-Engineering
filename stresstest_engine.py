import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from american_options_functions import (
    simulate_gbm, 
    lsm_train_max, 
    lsm_price_max
)

# ==============================================================================
# STRESSTEST ANALYSIS: VARIANCE CONVERGENCE AT LOW N (Variance=0,8)
# ==============================================================================

def run_n_convergence_stresstest(S0: float, K: float, T: float, r: float, 
                                 delta: float, sigma: float, m_steps: int, 
                                 n_grid: list, R_runs: int = 30):
    """
    Executes a stress test to demonstrate the numerical instability of Classic LSM 
    at low path counts (N) compared to the smoothed TVR Value Iteration.
    Calculates the variance over R independent macro-replications for each N.
    """
    print(f"\n--- Running Low-N Stresstest ({R_runs} Macro-Replications per N) ---")
    
    results = []
    
    for n_paths in n_grid:
        print(f"Testing N = {n_paths} paths...")
        
        lsm_prices = []
        tvr_prices = []
        
        for run in range(R_runs):
            # 1. Simulate independent paths for each run
            t1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
            t2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
            p1 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
            p2 = simulate_gbm(S0, sigma, r, delta, T, m_steps, n_paths)
            
            # 2. Classic LSM Bepreisung
            model_lsm = lsm_train_max(t1, t2, K, r, T, method="LSM")
            lsm_prices.append(lsm_price_max(model_lsm, p1, p2))
            
            # 3. TVR Value Iteration Bepreisung
            model_tvr = lsm_train_max(t1, t2, K, r, T, method="TVR")
            tvr_prices.append(lsm_price_max(model_tvr, p1, p2))
            
        # Berechne die empirische Varianz über die R Runs
        var_lsm = np.var(lsm_prices, ddof=1)
        var_tvr = np.var(tvr_prices, ddof=1)
        mean_lsm = np.mean(lsm_prices)
        mean_tvr = np.mean(tvr_prices)
        
        results.append({
            "N_Paths": n_paths,
            "LSM_Variance": var_lsm,
            "TVR_Variance": var_tvr,
            "LSM_Mean": mean_lsm,
            "TVR_Mean": mean_tvr
        })
        
    df_results = pd.DataFrame(results)
    
    # --- Generate LaTeX Table ---
    print("\nStresstest Results (Variance Comparison):")
    print(df_results.to_string(index=False))
    
    os.makedirs("output_tables", exist_ok=True)
    df_results.to_latex("output_tables/table_stresstest_variance.tex", index=False, float_format="%.4f")
    print("LaTeX table saved to 'output_tables/table_stresstest_variance.tex'")
    
    # --- Generate Convergence Plot ---
    os.makedirs("output_plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    plt.plot(df_results["N_Paths"], df_results["LSM_Variance"], 
             marker='o', linestyle='-', color='red', label='Classic LSM Variance')
    plt.plot(df_results["N_Paths"], df_results["TVR_Variance"], 
             marker='s', linestyle='-', color='blue', label='TVR Value Iteration Variance')
    
    plt.title("Stresstest: Estimator Variance at Low Path Counts ($N$)")
    plt.xlabel("Number of Simulated Paths ($N$)")
    plt.ylabel("Empirical Variance (over 30 runs)")
    
    # Log-Scale is often better to visualize explosive variance at the short end
    plt.yscale("log") 
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("output_plots/plot_stresstest_variance.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Plot saved to 'output_plots/plot_stresstest_variance.pdf'\n")
    
    return df_results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # We use the theoretical Glasserman parameters to isolate pure algorithmic performance
    S0_base = 100.0     
    K_val = 100.0       
    T_val = 3.0         
    r_val = 0.05        
    delta_val = 0.10    
    sigma_val = 0.80    
    m_steps = 9         
    
    # The Stresstest Grid: Focusing specifically on extremely low path counts
    # where the regression matrix becomes unstable for Classic LSM.
    path_grid = [250, 500, 1000, 2000, 4000]
    
    # 30 runs provide a statistically robust measurement of the variance
    num_macro_runs = 30 
    
    run_n_convergence_stresstest(
        S0=S0_base, K=K_val, T=T_val, r=r_val, delta=delta_val, 
        sigma=sigma_val, m_steps=m_steps, n_grid=path_grid, R_runs=num_macro_runs
    )