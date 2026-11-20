"""
Seed study for the classic LSM vs TVR comparison at S0 = 100 (stage-1 configuration).

Stage 1 of the project (commit 49747bd) reported TVR - LSM = +0.0294 (p = 0.0008) from one
50-replication run with the legacy global seed np.random.seed(2026). Rerunning the same experiment
with np.random.default_rng(2026) gave -0.0039 (p = 0.69). This script repeats the stage-1
configuration (N = 4,000 training and 4,000 pricing paths, R = 50, degree-2 basis, paths drawn
sequentially from one generator) for many seeds with both generators:

- legacy:  np.random.RandomState(seed), i.e. the MT19937 stream of np.random.seed(seed); with seed
           2026 the current code reproduces the stage-1 prices to 1e-15,
- current: np.random.default_rng(seed) (PCG64).

It reports the distribution of the 50-run paired mean difference TVR - LSM and where the two
single-run results fall in it. Both estimators are low-biased, so a positive difference means a
better TVR policy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from maxcall.lsm import lsm_price_max, lsm_train_max
from maxcall.mc_stats import df_to_markdown, fmt_p, paired_t, write_result
from maxcall.simulation import simulate_gbm

CFG = dict(S0=100.0, K=100.0, T=3.0, r=0.05, delta=0.10, sigma=0.20, m=9, N=4000, R=50)


def one_run(rng: np.random.Generator | np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """One stage-1 style experiment: R replications, paths drawn sequentially from `rng`."""
    c = CFG
    lsm, tvr = np.empty(c["R"]), np.empty(c["R"])
    for k in range(c["R"]):
        t1, t2, p1, p2 = (simulate_gbm(c["S0"], c["sigma"], c["r"], c["delta"], c["T"], c["m"],
                                       c["N"], rng) for _ in range(4))
        lsm[k] = lsm_price_max(lsm_train_max(t1, t2, c["K"], c["r"], c["T"], "LSM"), p1, p2)
        tvr[k] = lsm_price_max(lsm_train_max(t1, t2, c["K"], c["r"], c["T"], "TVR"), p1, p2)
    return lsm, tvr


def run_seed_study(seeds: range = range(1, 41)) -> pd.DataFrame:
    """Runs the stage-1 experiment for every seed with both generators and writes the summary."""
    makers = {"legacy MT19937 (np.random.seed)": np.random.RandomState,
              "PCG64 (np.random.default_rng)": np.random.default_rng}
    rows, pooled, means = [], {}, {}
    for name, make in makers.items():
        diffs, run_means, sig = [], [], 0
        for s in seeds:
            lsm, tvr = one_run(make(s))
            t = paired_t(tvr, lsm)
            diffs.append(tvr - lsm)
            run_means.append(t["diff"])
            sig += int(t["p"] < 0.05 and t["diff"] > 0)
        run_means = np.array(run_means)
        d = np.concatenate(diffs)
        pooled[name], means[name] = d, run_means
        rows.append({"Generator": name, "Runs": len(run_means),
                     "Mean of run means": run_means.mean(), "SD across runs": run_means.std(ddof=1),
                     "Min": run_means.min(), "Max": run_means.max(),
                     "Runs with p < 0.05 (TVR better)": sig, "Runs with negative mean": int(np.sum(run_means < 0)),
                     "Pooled mean": d.mean(), "Pooled SE": d.std(ddof=1) / np.sqrt(len(d))})
    df = pd.DataFrame(rows)
    a, b = means.values()
    welch = stats.ttest_ind(a, b, equal_var=False)
    allruns = np.concatenate([a, b])
    allreps = np.concatenate(list(pooled.values()))
    mu, sd = allreps.mean(), allruns.std(ddof=1)
    se_mu = allreps.std(ddof=1) / np.sqrt(len(allreps))

    # The two single runs being reconciled
    single = []
    for name, rng in (("Stage 1: np.random.seed(2026)", np.random.RandomState(2026)),
                      ("Rerun: np.random.default_rng(2026)", np.random.default_rng(2026))):
        t = paired_t(*one_run(rng)[::-1])
        single.append({"Run": name, "TVR - LSM": t["diff"], "SE": t["se"], "p": t["p"],
                       "z in seed distribution": (t["diff"] - mu) / sd,
                       "Share of study runs at least as extreme": float(
                           np.mean(allruns >= t["diff"]) if t["diff"] > mu else np.mean(allruns <= t["diff"]))})
    df_single = pd.DataFrame(single)
    print(df.to_string(index=False))
    print(df_single.to_string(index=False))
    print(f"Generators differ? Welch p = {welch.pvalue:.3f}. Pooled over both: {mu:.4f} (SE {se_mu:.4f})")

    fmt = {"Mean of run means": "{:+.4f}", "SD across runs": "{:.4f}", "Min": "{:+.4f}",
           "Max": "{:+.4f}", "Pooled mean": "{:+.4f}", "Pooled SE": "{:.4f}", "TVR - LSM": "{:+.4f}",
           "SE": "{:.4f}", "p": fmt_p, "z in seed distribution": "{:+.2f}",
           "Share of study runs at least as extreme": "{:.3f}"}
    c = CFG
    text = (f"# Seed study: classic LSM vs TVR at S0 = 100 (stage-1 configuration)\n\n"
            f"Each run: R = {c['R']} replications, N = {c['N']:,} training and {c['N']:,} pricing "
            f"paths drawn sequentially from one generator, degree-2 basis, both methods on the "
            f"same paths; statistic = paired mean difference TVR - LSM. Seeds {seeds.start}.."
            f"{seeds.stop - 1} for each generator. With the legacy generator and seed 2026 the "
            f"current code reproduces the stage-1 code (commit 49747bd) to 1e-15. Generated by "
            f"`seed_study.py`.\n\n"
            + df_to_markdown(df, fmt)
            + f"\n\nThe two generators give the same distribution (Welch test on the run means, "
              f"p = {welch.pvalue:.2f}). Pooled over both: TVR - LSM = {mu:+.4f} (SE {se_mu:.4f}, "
              f"{len(allreps):,} replications); SD of a 50-run mean {sd:.4f}.\n\n"
              "The two single runs in this distribution:\n\n"
            + df_to_markdown(df_single, fmt) + "\n")
    write_result("seed_study", text)
    return df


def main() -> None:
    """Runs the seed study and writes results/seed_study.md."""
    t0 = time.perf_counter()
    run_seed_study()
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
