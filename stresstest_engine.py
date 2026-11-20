"""
Stress test: estimator variance at low path counts and high volatility (sigma = 0.8), otherwise
Glasserman benchmark parameters (S0 = K = 100, T = 3, r = 0.05, delta = 0.10, 9 exercise dates).

For each N (training paths = pricing paths = N) and replication k, all methods are trained and
priced on the same paths (common random numbers, generator default_rng([seed, 8, N, k])):
the four methods of the 2x2 design with the default scaled basis, and classic LSM with the raw
(unscaled) basis to isolate the effect of scaling. Every price is low-biased. A reference interval
for the true price comes from the duality bounds of a TVR policy trained on 100,000 paths.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from american_options_functions import (METHODS, DEFAULT_DEGREE, lsm_price_max, lsm_train_max,
                                        simulate_gbm)
from duality import duality_bounds
from mc_stats import df_to_markdown, fmt_p, holm, paired_t, pitman_morgan, summarize, write_result

STRESS = dict(S0=100.0, K=100.0, T=3.0, r=0.05, delta=0.10, sigma=0.80, m=9)
CONFIGS = [("LSM", True), ("LSM-all", True), ("TVR-ITM", True), ("TVR", True), ("LSM", False)]


def label(cfg: tuple[str, bool]) -> str:
    """Short label of a (method, scaled) configuration."""
    return cfg[0] + ("" if cfg[1] else " (raw basis)")


def reference_interval(seed: int = 2026) -> dict:
    """Duality bounds at the stress parameters for a TVR policy trained on 100,000 paths."""
    p = STRESS
    rng = np.random.default_rng([seed, 9])
    t1 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], 100_000, rng)
    t2 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], 100_000, rng)
    model = lsm_train_max(t1, t2, p["K"], p["r"], p["T"], method="TVR")
    return duality_bounds(model, p["S0"], p["delta"], p["sigma"], p["sigma"], 0.0,
                          n_lower=200_000, n_outer=4000, n_inner=2000, rng=rng)


def run_n_convergence_stresstest(n_grid: tuple[int, ...] = (250, 500, 1000, 2000, 4000),
                                 R: int = 50, seed: int = 2026) -> tuple[pd.DataFrame, ...]:
    """
    Mean, SE, variance over R replications and the median / max condition number of the
    regression matrices, for every configuration and N. Variance comparison classic LSM vs TVR
    by the Pitman-Morgan test for paired samples (Holm over the N grid), mean comparison by
    paired t-tests (Holm over the N grid), and the largest absolute price difference between
    the raw and the scaled basis for classic LSM.
    """
    p = STRESS
    rows, test_rows = [], []
    for N in n_grid:
        prices = {c: np.empty(R) for c in CONFIGS}
        conds = {c: [] for c in CONFIGS}
        for k in range(R):
            rng = np.random.default_rng([seed, 8, N, k])
            t1 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], N, rng)
            t2 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], N, rng)
            p1 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], N, rng)
            p2 = simulate_gbm(p["S0"], p["sigma"], p["r"], p["delta"], p["T"], p["m"], N, rng)
            for c in CONFIGS:
                model = lsm_train_max(t1, t2, p["K"], p["r"], p["T"], method=c[0],
                                      degree=DEFAULT_DEGREE, scale=c[1], diagnostics=True)
                prices[c][k] = lsm_price_max(model, p1, p2)
                conds[c].extend(model["cond"].values())
        for c in CONFIGS:
            st = summarize(prices[c])
            lc = np.log10(conds[c])
            rows.append({"N": N, "Method": label(c), "Mean": st["mean"], "SE": st["se"],
                         "Variance": st["var"], "log10 cond (median)": float(np.median(lc)),
                         "log10 cond (max)": float(np.max(lc))})
        lsm, tvr = prices[("LSM", True)], prices[("TVR", True)]
        pm = pitman_morgan(lsm, tvr)
        t = paired_t(tvr, lsm)
        raw_diff = float(np.max(np.abs(prices[("LSM", False)] - prices[("LSM", True)])))
        test_rows.append({"N": N, "Var LSM / Var TVR": pm["var_ratio"], "Pitman-Morgan p": pm["p"],
                          "Mean TVR - LSM": t["diff"], "SE (diff)": t["se"], "Paired t p": t["p"],
                          "max abs. LSM raw - scaled": raw_diff})
        print(f"N = {N:5d}: " + " | ".join(
            f"{label(c)} {np.mean(prices[c]):.3f} (var {np.var(prices[c], ddof=1):.2f})"
            for c in CONFIGS))
    df = pd.DataFrame(rows)
    df_t = pd.DataFrame(test_rows)
    df_t.insert(3, "PM p (Holm)", holm(df_t["Pitman-Morgan p"].values))
    df_t.insert(7, "t p (Holm)", holm(df_t["Paired t p"].values))
    return df, df_t


def plot_variance(df: pd.DataFrame, R: int) -> None:
    """Variance of the R price estimates vs N (log scale) for the four scaled methods."""
    plt.figure(figsize=(8, 5))
    styles = {"LSM": ("s-", "tab:blue"), "LSM-all": ("^-", "tab:green"),
              "TVR-ITM": ("v--", "tab:purple"), "TVR": ("o-", "tab:red")}
    for m in METHODS:
        d = df[df["Method"] == m]
        plt.plot(d["N"], d["Variance"], styles[m][0], color=styles[m][1], label=m)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of training and pricing paths N")
    plt.ylabel(f"Variance of the price over {R} replications")
    plt.title("Stress test (σ = 0.8): estimator variance vs N")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("output_plots", exist_ok=True)
    plt.savefig("output_plots/stresstest_variance.png", dpi=130)
    plt.savefig("results/stresstest_variance.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    t0 = time.perf_counter()
    R = 50
    df, df_t = run_n_convergence_stresstest(R=R)
    ref = reference_interval()
    print(df.to_string(index=False))
    print(df_t.to_string(index=False))
    print(f"Reference interval (TVR, 100,000 training paths): L = {ref['L']:.4f} "
          f"({ref['se_L']:.4f}), U = {ref['U']:.4f} ({ref['se_U']:.4f}), "
          f"95% [{ref['ci_low']:.4f}, {ref['ci_high']:.4f}]")
    plot_variance(df, R)

    fmt = {"N": "{:.0f}", "Mean": "{:.3f}", "SE": "{:.3f}", "Variance": "{:.3f}",
           "log10 cond (median)": "{:.1f}", "log10 cond (max)": "{:.1f}",
           "Var LSM / Var TVR": "{:.3f}", "Pitman-Morgan p": fmt_p, "PM p (Holm)": fmt_p,
           "Mean TVR - LSM": "{:+.3f}", "SE (diff)": "{:.3f}", "Paired t p": fmt_p,
           "t p (Holm)": fmt_p, "max abs. LSM raw - scaled": "{:.2e}"}
    df_out = df.copy()
    df_out["N"] = df_out["N"].astype(int)
    df_t_out = df_t.copy()
    text = (f"# Stress test: low path counts at sigma = 0.8\n\n"
            f"S0 = K = 100, T = 3, r = 0.05, delta = 0.10, sigma = 0.8, rho = 0, 9 exercise dates. "
            f"For each N: R = {R} replications with N training and N independent pricing paths; "
            f"all configurations use the same paths (seed 2026). Basis degree {DEFAULT_DEGREE}; "
            f"'raw basis' = classic LSM without scaling by K. Condition numbers over all dates "
            f"and replications. Generated by `stresstest_engine.py`.\n\n"
            f"Reference interval for the true price (duality bounds of a TVR policy trained on "
            f"100,000 paths; 200,000 lower-bound paths, 4,000 x 2,000 nested paths): "
            f"L = {ref['L']:.3f} (SE {ref['se_L']:.3f}), U = {ref['U']:.3f} (SE {ref['se_U']:.3f}), "
            f"95% interval [{ref['ci_low']:.3f}, {ref['ci_high']:.3f}].\n\n"
            + df_to_markdown(df_out, fmt)
            + "\n\nClassic LSM vs TVR (scaled basis) per N, Holm correction over the five N values "
              "for each test, and the effect of scaling on classic LSM:\n\n"
            + df_to_markdown(df_t_out, fmt)
            + "\n\n![Variance vs N](stresstest_variance.png)\n")
    write_result("stresstest", text)
    os.makedirs("output_tables", exist_ok=True)
    df.to_csv("output_tables/stresstest.csv", index=False)
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")
