"""
Price bounds for the Glasserman benchmark (Bermudan max-call, Example 8.6.1): lower bound from
the out-of-sample forward pass of a learned policy, upper bound from the Andersen-Broadie
duality gap of the same policy (nested simulation). See duality.py for the construction.

For each S0 several policies are evaluated on common random numbers (same lower-bound, outer and
inner paths), so their lower bounds and gaps can be compared pairwise.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from maxcall.lsm import lsm_train_max
from maxcall.simulation import simulate_gbm
from maxcall.duality import duality_bounds
from maxcall.mc_stats import paired_t, holm, df_to_markdown, write_result, fmt_p

BENCH = dict(K=100.0, T=3.0, r=0.05, delta=0.10, sigma=0.20, m=9)
REFERENCE = {90.0: 8.08, 100.0: 13.90, 110.0: 21.34}

# (label, method, number of training paths); the first entry is the default policy.
POLICIES = [
    ("TVR, N_train = 4,000 (default)", "TVR", 4000),
    ("Classic LSM, N_train = 4,000", "LSM", 4000),
    ("TVR, N_train = 500", "TVR", 500),
    ("TVR, N_train = 100,000", "TVR", 100_000),
]


def run_bounds(S0_values: tuple[float, ...] = (90.0, 100.0, 110.0), n_lower: int = 200_000,
               n_outer: int = 4000, n_inner: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """
    Lower bound L, upper bound U, SEs, gap U - L and the 95% interval
    [L - 1.96 SE_L, U + 1.96 SE_U] for each S0 and policy. Training paths come from
    default_rng([seed, 4, S0]) (the 500-path policy uses the first 500 of the 4,000 paths);
    bounds use default_rng([seed, 5, S0]) for every policy (common random numbers).
    """
    b = BENCH
    rows, paired_rows = [], []
    for S0 in S0_values:
        rng_tr = np.random.default_rng([seed, 4, int(S0)])
        t1 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], 4000, rng_tr)
        t2 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], 4000, rng_tr)
        big1 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], 100_000, rng_tr)
        big2 = simulate_gbm(S0, b["sigma"], b["r"], b["delta"], b["T"], b["m"], 100_000, rng_tr)
        res = {}
        for label, method, n_train in POLICIES:
            if n_train == 100_000:
                tr1, tr2 = big1, big2
            else:
                tr1, tr2 = t1[:n_train], t2[:n_train]
            model = lsm_train_max(tr1, tr2, b["K"], b["r"], b["T"], method=method)
            out = duality_bounds(model, S0, b["delta"], b["sigma"], b["sigma"], 0.0,
                                 n_lower, n_outer, n_inner, np.random.default_rng([seed, 5, int(S0)]))
            res[label] = out
            ref = REFERENCE[S0]
            rows.append({"S0": S0, "Policy": label, "L": out["L"], "SE L": out["se_L"],
                         "U": out["U"], "SE U": out["se_U"], "Gap U - L": out["gap"],
                         "SE gap": out["se_gap"], "95% low": out["ci_low"],
                         "95% high": out["ci_high"], "Reference": ref,
                         "Ref in interval": bool(out["ci_low"] <= ref <= out["ci_high"]),
                         "L <= U": bool(out["L"] <= out["U"]), "Seconds": out["seconds"]})
            print(f"S0 = {S0:5.1f} | {label:32s} | L = {out['L']:.4f} ({out['se_L']:.4f}) | "
                  f"U = {out['U']:.4f} ({out['se_U']:.4f}) | gap = {out['gap']:.4f} "
                  f"({out['se_gap']:.4f}) | {out['seconds']:.1f} s")
        # Paired comparisons against the default policy (common random numbers)
        base = res[POLICIES[0][0]]
        for label, _, _ in POLICIES[1:]:
            o = res[label]
            dL = paired_t(o["lower_paths"], base["lower_paths"])
            dG = paired_t(o["gap_paths"], base["gap_paths"])
            paired_rows.append({"S0": S0, "Policy": label, "L - L_default": dL["diff"],
                                "SE (L)": dL["se"], "p (L)": dL["p"],
                                "Gap - gap_default": dG["diff"], "SE (gap)": dG["se"],
                                "p (gap)": dG["p"]})
    df = pd.DataFrame(rows)
    df_pair = pd.DataFrame(paired_rows)
    df_pair.insert(df_pair.columns.get_loc("p (L)") + 1, "Holm p (L)", holm(df_pair["p (L)"].values))
    df_pair["Holm p (gap)"] = holm(df_pair["p (gap)"].values)
    print(df.to_string(index=False))
    print(df_pair.to_string(index=False))

    fmt = {"S0": "{:.0f}", "L": "{:.4f}", "SE L": "{:.4f}", "U": "{:.4f}", "SE U": "{:.4f}",
           "Gap U - L": "{:.4f}", "SE gap": "{:.4f}", "95% low": "{:.4f}", "95% high": "{:.4f}",
           "Reference": "{:.2f}", "Seconds": "{:.1f}", "L - L_default": "{:+.4f}",
           "SE (L)": "{:.4f}", "p (L)": fmt_p, "Gap - gap_default": "{:+.4f}",
           "SE (gap)": "{:.4f}", "p (gap)": fmt_p,
           "Holm p (L)": fmt_p, "Holm p (gap)": fmt_p}
    text = (f"# Lower and upper bounds (Andersen-Broadie duality)\n\n"
            f"Glasserman benchmark, basis degree 2 scaled by K. Lower bound L: forward pass of the "
            f"policy on {n_lower:,} independent paths. Upper bound U = L + gap, gap estimated "
            f"with {n_outer:,} outer paths and {n_inner:,} inner paths per continuation value "
            f"(inner simulations only at in-the-money dates). 95% interval "
            f"[L - 1.96 SE_L, U + 1.96 SE_U]. All policies at one S0 use the same lower-bound, "
            f"outer and inner random numbers (seed {seed}). Generated by `bounds_engine.py`.\n\n"
            + df_to_markdown(df.drop(columns=["Seconds"]), fmt)
            + "\n\nPaired differences against the default policy (same paths; positive "
              "L - L_default means a better policy). Holm correction over the 9 L-tests and, "
              "separately, the 9 gap-tests. These compare the specific trained policies "
              "(one training set per S0), conditional on their training paths:\n\n"
            + df_to_markdown(df_pair, fmt) + "\n")
    write_result("duality_bounds", text)
    os.makedirs("output_tables", exist_ok=True)
    df.to_csv("output_tables/duality_bounds.csv", index=False)
    return df


def main() -> None:
    """Runs the full analysis of this script and writes results/ and output_*/."""
    t0 = time.perf_counter()
    run_bounds()
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
