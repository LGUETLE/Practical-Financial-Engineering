"""
Calibrated scenarios: max-call on two assets with volatilities and correlation estimated from
daily Yahoo Finance prices (AAPL/MSFT and BTC-USD/ETH-USD, same window 2022-01-01 to 2025-12-31).

Contract: S0 = 100 for both assets (normalised), K = 100, T = 1, r = 0.05, 12 exercise dates and
dividend yield delta = 0. With delta = 0 the discounted price e^{-rt} S_j(t) is a martingale, so
e^{-rt} max(S_1(t), S_2(t)) is a submartingale under Q and so is the discounted payoff
e^{-rt} (max(S_1, S_2) - K)^+ (convex function of a submartingale, K e^{-rt} decreasing). Early
exercise is therefore never optimal and the Bermudan price equals the European price, for which
the Stulz (1982) closed form is exact. This gives an exact reference for the bias of every
regression method: any early exercise by a learned policy can only lose value.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from american_options_functions import (METHODS, apply_policy, lsm_train_max,
                                        simulate_correlated_gbm)
from data_fetcher import HistoricalDataFetcher
from european import stulz_max_call
from mc_stats import df_to_markdown, fmt_p, holm, paired_t, summarize, write_result

CONTRACT = dict(S0=100.0, K=100.0, r=0.05, T=1.0, m=12, delta=0.0)
SCENARIOS = [("tech", "AAPL", "MSFT", "stock"), ("crypto", "BTC-USD", "ETH-USD", "crypto")]
METHOD_NAMES = {"LSM": "Classic LSM (cash flow, ITM)", "LSM-all": "Cash flow, all paths",
                "TVR-ITM": "Value iteration, ITM", "TVR": "TVR (value iteration, all)"}


def price_methods(sigma_1: float, sigma_2: float, rho: float, n_paths: int,
                  rng: np.random.Generator) -> dict:
    """
    One replication: simulate training and independent pricing paths once and train/price all
    four methods on them (common random numbers). Returns {method: (price, early-exercise share)},
    where the early-exercise share is the fraction of pricing paths stopped before maturity.
    """
    c = CONTRACT
    t1, t2 = simulate_correlated_gbm(c["S0"], c["S0"], sigma_1, sigma_2, rho, c["r"], c["delta"],
                                     c["T"], c["m"], n_paths, rng)
    p1, p2 = simulate_correlated_gbm(c["S0"], c["S0"], sigma_1, sigma_2, rho, c["r"], c["delta"],
                                     c["T"], c["m"], n_paths, rng)
    dt = c["T"] / c["m"]
    out = {}
    for meth in METHODS:
        model = lsm_train_max(t1, t2, c["K"], c["r"], c["T"], method=meth)
        payoff, ex_idx = apply_policy(model, p1, p2)
        disc = payoff * np.exp(-c["r"] * dt * ex_idx)
        out[meth] = (float(disc.mean()), float(np.mean(ex_idx < c["m"])))
    return out


def run_empirical_scenario(name: str, params: dict, R: int = 50, n_paths: int = 10_000,
                           seed: int = 2026, scenario_id: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    R replications (generator default_rng([seed, 6, scenario_id, k])) of all four methods on the
    calibrated parameters. Reports per method mean, SE, 95% CI, bias against the Stulz price,
    RMSE, bias / SE, the two-sided p-value of H0: bias = 0 (Holm correction applied by the
    caller) and the mean share of pricing paths exercised early.
    """
    c = CONTRACT
    s1, s2, rho = params["sigma_1"], params["sigma_2"], params["rho"]
    stulz = stulz_max_call(c["S0"], c["S0"], c["K"], c["r"], c["T"], c["delta"], c["delta"],
                           s1, s2, rho)
    print(f"\n--- {name.upper()}: R = {R}, N = {n_paths:,}, Stulz price {stulz:.4f} ---")
    prices = {m: np.empty(R) for m in METHODS}
    early = {m: np.empty(R) for m in METHODS}
    for k in range(R):
        res = price_methods(s1, s2, rho, n_paths, np.random.default_rng([seed, 6, scenario_id, k]))
        for m, (p, e) in res.items():
            prices[m][k], early[m][k] = p, e
    rows = []
    for m in METHODS:
        st = summarize(prices[m], stulz)
        t = st["bias"] / st["se"]
        rows.append({"Scenario": name, "Method": METHOD_NAMES[m], "Mean": st["mean"],
                     "SE": st["se"], "CI low": st["ci_low"], "CI high": st["ci_high"],
                     "Stulz": stulz, "Bias": st["bias"], "Bias / SE": t,
                     "p (bias = 0)": float(2 * stats.t.sf(abs(t), R - 1)), "RMSE": st["rmse"],
                     "Early exercise share": float(early[m].mean())})
    df = pd.DataFrame(rows)

    os.makedirs("output_plots", exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.boxplot([prices[m] for m in METHODS], tick_labels=list(METHODS))
    plt.axhline(stulz, color="black", ls="--", lw=1, label=f"Stulz (exact) = {stulz:.3f}")
    plt.title(f"Price estimates over {R} replications: {name}")
    plt.ylabel("Price estimate")
    plt.legend()
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output_plots/boxplot_{name}.png", dpi=130)
    plt.close()
    return df, prices


def run_sigma_sensitivity(tech_vol: float, crypto_vol: float,
                          sigmas: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
                          rho: float = 0.5, R: int = 20, n_paths: int = 10_000,
                          seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Price and bias against the Stulz price as a function of a common volatility
    sigma_1 = sigma_2 = sigma (rho = 0.5 fixed), for all four methods. R replications per sigma
    (generator default_rng([seed, 7, round(100 sigma), k])), common random numbers across methods.
    Plot: price vs sigma with the exact European price, and bias with 95% confidence intervals.
    Also returns paired contrasts per sigma (TVR - classic LSM, target effect, sample effect as
    in the 2x2 design) with Holm correction over all of them.
    """
    print(f"\n--- Volatility sensitivity (rho = {rho}, R = {R}, N = {n_paths:,}) ---")
    c = CONTRACT
    rows, con_rows = [], []
    for sig in sigmas:
        stulz = stulz_max_call(c["S0"], c["S0"], c["K"], c["r"], c["T"], 0.0, 0.0, sig, sig, rho)
        prices = {m: np.empty(R) for m in METHODS}
        for k in range(R):
            res = price_methods(sig, sig, rho, n_paths,
                                np.random.default_rng([seed, 7, int(round(sig * 100)), k]))
            for m, (p, _) in res.items():
                prices[m][k] = p
        for m in METHODS:
            st = summarize(prices[m], stulz)
            rows.append({"sigma": sig, "Method": m, "Mean": st["mean"], "SE": st["se"],
                         "Stulz": stulz, "Bias": st["bias"], "RMSE": st["rmse"]})
        cf_itm, cf_all, v_itm, v_all = (prices[m] for m in ("LSM", "LSM-all", "TVR-ITM", "TVR"))
        zero = np.zeros(R)
        for name, d in (("TVR - classic LSM", v_all - cf_itm),
                        ("Target (value - cash flow)", ((v_itm - cf_itm) + (v_all - cf_all)) / 2),
                        ("Sample (all - ITM)", ((cf_all - cf_itm) + (v_all - v_itm)) / 2)):
            t = paired_t(d, zero)
            con_rows.append({"sigma": sig, "Contrast": name, "Estimate": t["diff"], "SE": t["se"],
                             "p": t["p"]})
        print(f"  sigma = {sig:.2f}: Stulz {stulz:.4f} | " + " | ".join(
            f"{m} {np.mean(prices[m]) - stulz:+.4f}" for m in METHODS))
    df = pd.DataFrame(rows)
    df_con = pd.DataFrame(con_rows)
    df_con["p (Holm)"] = holm(df_con["p"].values)
    print(df_con.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)
    styles = {"LSM": ("s--", "tab:blue"), "LSM-all": ("^-", "tab:green"),
              "TVR-ITM": ("v--", "tab:purple"), "TVR": ("o-", "tab:red")}
    ax = axes[0]
    ref = df[df["Method"] == "TVR"]
    ax.plot(ref["sigma"], ref["Stulz"], "k-", lw=2.5, alpha=0.4, label="Stulz (exact)")
    for m in METHODS:
        d = df[df["Method"] == m]
        ax.plot(d["sigma"], d["Mean"], styles[m][0], color=styles[m][1], ms=4, label=m)
    ax.set(xlabel="Volatility σ (both assets)", ylabel="Price", title="Price vs volatility (ρ = 0.5)")
    ax.legend(fontsize=8)
    ax = axes[1]
    width = 0.012
    for j, m in enumerate(METHODS):
        d = df[df["Method"] == m]
        ax.errorbar(d["sigma"] + (j - 1.5) * width, d["Bias"], yerr=1.96 * d["SE"],
                    fmt=styles[m][0][0], color=styles[m][1], capsize=2, ms=4, label=m)
    ax.axhline(0, color="black", lw=0.8)
    for v, lab, col in ((tech_vol, "tech", "darkgreen"), (crypto_vol, "crypto", "darkorange")):
        ax.axvline(v, color=col, ls="-.", lw=1.2, label=f"{lab} avg. σ ≈ {v:.2f}")
    ax.set(xlabel="Volatility σ (both assets)", ylabel="Mean price − Stulz price",
           title=f"Bias with 95% CI (R = {R} per σ)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    os.makedirs("output_plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    plt.savefig("output_plots/empirical_sigma_sensitivity.png", bbox_inches="tight")
    plt.savefig("results/empirical_sigma_sensitivity.png", bbox_inches="tight")
    plt.close()
    return df, df_con


if __name__ == "__main__":
    t_start = time.perf_counter()
    c = CONTRACT
    calib, tables, all_prices = {}, [], {}
    for j, (name, tk1, tk2, cls) in enumerate(SCENARIOS):
        fetcher = HistoricalDataFetcher(tk1, tk2, cls)
        params = fetcher.calibrate(fetcher.fetch_data())
        calib[name] = dict(params, tickers=f"{tk1} / {tk2}")
        df, pr = run_empirical_scenario(name, params, scenario_id=j)
        tables.append(df)
        all_prices[name] = pr
    df_all = pd.concat(tables, ignore_index=True)
    df_all["p (Holm)"] = holm(df_all["p (bias = 0)"].values)
    print(df_all.to_string(index=False))
    stulz_ok = bool(np.all(df_all["CI low"] <= df_all["Stulz"]))
    print(f"Check: Stulz price above every mean or within its 95% CI: {'PASS' if stulz_ok else 'FAIL'}")

    df_cal = pd.DataFrame([{"Scenario": k, "Assets": v["tickers"], "Window": f"{v['first_date']} to "
                            f"{v['last_date']}", "Daily returns": v["n_returns"],
                            "Periods/year": v["periods_per_year"], "sigma_1": v["sigma_1"],
                            "sigma_2": v["sigma_2"], "rho": v["rho"]} for k, v in calib.items()])
    fmt = {"Mean": "{:.4f}", "SE": "{:.4f}", "CI low": "{:.4f}", "CI high": "{:.4f}",
           "Stulz": "{:.4f}", "Bias": "{:+.4f}", "Bias / SE": "{:+.2f}", "p (bias = 0)": fmt_p,
           "p (Holm)": fmt_p, "RMSE": "{:.4f}", "Early exercise share": "{:.4f}",
           "sigma_1": "{:.4f}", "sigma_2": "{:.4f}", "rho": "{:.4f}"}
    avg_vol = {k: (v["sigma_1"] + v["sigma_2"]) / 2 for k, v in calib.items()}
    df_sens, df_con = run_sigma_sensitivity(avg_vol["tech"], avg_vol["crypto"])

    text = ("# Calibrated scenarios: bias against the exact European price\n\n"
            "Daily adjusted closes from Yahoo Finance, identical window for both asset classes. "
            "Only volatilities and correlation are calibrated (log-returns, annualised with 252 "
            "trading days for stocks and 365 days for crypto); pricing uses the risk-neutral "
            "drift r. Contract: S0 = 100 for both assets (normalised), K = 100, T = 1, r = 0.05, "
            "12 exercise dates, dividend yield 0. Without dividends early exercise of a max-call "
            "is never optimal, so the Bermudan price equals the European price (Stulz closed "
            "form), an exact reference. Generated by `empirical_engine.py`.\n\n"
            + df_to_markdown(df_cal, fmt)
            + "\n\nR = 50 replications, N = 10,000 training and 10,000 independent pricing paths, "
              "all four methods on the same paths (seed 2026). Bias = mean - Stulz; p-value of "
              "H0: bias = 0 (t-test over replications) with Holm correction over all 8 tests. "
              "Early exercise share = fraction of pricing paths that the learned policy stops "
              "before maturity (optimal: 0).\n\n"
            + df_to_markdown(df_all.drop(columns=["CI low", "CI high"]), fmt)
            + "\n\nVolatility sensitivity (sigma_1 = sigma_2 = sigma, rho = 0.5, R = 20 per sigma, "
              "N = 10,000): bias against the Stulz price.\n\n"
            + df_to_markdown(df_sens.pivot_table(index="sigma", columns="Method", values="Bias")
                             .reset_index()[["sigma"] + list(METHODS)],
                             {"sigma": "{:.1f}", **{m: "{:+.4f}" for m in METHODS}})
            + "\n\nStandard errors of these biases (same layout):\n\n"
            + df_to_markdown(df_sens.pivot_table(index="sigma", columns="Method", values="SE")
                             .reset_index()[["sigma"] + list(METHODS)],
                             {"sigma": "{:.1f}", **{m: "{:.4f}" for m in METHODS}})
            + "\n\nPaired contrasts per sigma (same paths for all methods; positive = higher, "
              "i.e. less biased, price). Holm correction over all 21 tests:\n\n"
            + df_to_markdown(df_con, {"sigma": "{:.1f}", "Estimate": "{:+.4f}", "SE": "{:.4f}",
                                      "p": fmt_p, "p (Holm)": fmt_p})
            + "\n\n![Volatility sensitivity](empirical_sigma_sensitivity.png)\n")
    write_result("empirical", text)
    os.makedirs("output_tables", exist_ok=True)
    df_all.to_csv("output_tables/empirical_bias.csv", index=False)
    df_sens.to_csv("output_tables/empirical_sigma_sensitivity.csv", index=False)
    print(f"\nTotal runtime: {time.perf_counter() - t_start:.1f} s")
