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
    lsm_price_max_cv,
    DEFAULT_DEGREE,
    DEFAULT_SCALE,
)
from european import stulz_max_call, mc_max_call
from mc_stats import (summarize, paired_t, pitman_morgan, holm, df_to_markdown,
                      write_result, fmt_p)

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
                                      "t": "{:.3f}", "p": fmt_p, "p (Holm)": fmt_p})
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


METHOD_LABEL = {"LSM": "Classic LSM (cash flow, ITM)", "LSM-all": "Cash flow, all paths",
                "TVR-ITM": "Value iteration, ITM", "TVR": "TVR (value iteration, all)"}


def analyze_2x2(prices: dict, S0_values: tuple[float, ...] = (90.0, 100.0, 110.0),
                R: int = 50, n_paths: int = 4000, seed: int = 2026,
                confirm: dict | None = None) -> dict:
    """
    Analysis of the 2x2 design (target {cash flow, value} x sample {ITM, all}) on the replicated
    benchmark. All methods are low-biased and priced on the same paths, so for each S0
      (a) the method with the highest mean is the best policy; the other three are compared
          with it by paired t-tests,
      (b) the factorial contrasts per replication are tested with paired t-tests:
          target effect      = [(V,ITM) - (CF,ITM) + (V,all) - (CF,all)] / 2
          sample effect      = [(CF,all) - (CF,ITM) + (V,all) - (V,ITM)] / 2
          interaction        = (V,all) - (V,ITM) - (CF,all) + (CF,ITM).
    Holm correction is applied over all tests of (a) together (9) and of (b) together (9).
    """
    methods = list(METHOD_LABEL)
    df_sum = summarize_benchmark(prices, tuple(methods))
    df_sum.insert(2, "Target", [("cash flow" if m.startswith("LSM") else "value") for m in df_sum["Method"]])
    df_sum.insert(3, "Sample", [("ITM" if m in ("LSM", "TVR-ITM") else "all") for m in df_sum["Method"]])

    best_rows, contrast_rows = [], []
    for S0 in S0_values:
        means = {m: float(np.mean(prices[(S0, m)])) for m in methods}
        best = max(means, key=means.get)
        for m in methods:
            if m == best:
                continue
            t = paired_t(prices[(S0, m)], prices[(S0, best)])
            best_rows.append({"S0": S0, "Best": best, "Method": m, "Diff vs best": t["diff"],
                              "SE": t["se"], "t": t["t"], "p": t["p"]})
        cf_itm, cf_all = prices[(S0, "LSM")], prices[(S0, "LSM-all")]
        v_itm, v_all = prices[(S0, "TVR-ITM")], prices[(S0, "TVR")]
        zero = np.zeros(len(cf_itm))
        contrasts = {
            "Target (value - cash flow)": ((v_itm - cf_itm) + (v_all - cf_all)) / 2,
            "Sample (all - ITM)": ((cf_all - cf_itm) + (v_all - v_itm)) / 2,
            "Interaction": (v_all - v_itm) - (cf_all - cf_itm),
        }
        for name, c in contrasts.items():
            t = paired_t(c, zero)
            contrast_rows.append({"S0": S0, "Effect": name, "Estimate": t["diff"], "SE": t["se"],
                                  "t": t["t"], "p": t["p"]})
    df_best = pd.DataFrame(best_rows)
    df_best["p (Holm)"] = holm(df_best["p"].values)
    df_con = pd.DataFrame(contrast_rows)
    df_con["p (Holm)"] = holm(df_con["p"].values)

    print("\n2x2 design: summary")
    print(df_sum.to_string(index=False))
    print("\nPaired differences against the best method per S0 (Holm over all 9 tests)")
    print(df_best.to_string(index=False))
    print("\nFactorial contrasts (Holm over all 9 tests)")
    print(df_con.to_string(index=False))

    fmt = {"Mean": "{:.4f}", "SE": "{:.4f}", "CI low": "{:.4f}", "CI high": "{:.4f}",
           "Reference": "{:.2f}", "Bias": "{:+.4f}", "RMSE": "{:.4f}", "S0": "{:.0f}",
           "Diff vs best": "{:+.4f}", "t": "{:.2f}", "p": fmt_p, "p (Holm)": fmt_p,
           "Estimate": "{:+.4f}"}
    text = (f"# 2x2 design: regression target x regression sample (Glasserman benchmark)\n\n"
            f"R = {R} replications per S0, N = {n_paths:,} training and {n_paths:,} independent "
            f"pricing paths. In each replication all four methods use the same training and "
            f"pricing paths, and the same normals are used for all S0 (common random numbers, "
            f"seed {seed}). Basis: degree {DEFAULT_DEGREE}, scaled by K = {DEFAULT_SCALE}. "
            f"SE = standard error over replications; bias and RMSE against the Glasserman "
            f"reference. Generated by `theory_engine.py`.\n\n"
            + df_to_markdown(df_sum, fmt)
            + "\n\nPaired differences against the method with the highest mean at each S0 "
              "(all four estimators are low-biased, so a negative difference means a worse "
              "policy). Holm correction over all 9 tests:\n\n"
            + df_to_markdown(df_best, fmt)
            + "\n\nFactorial contrasts per S0 (paired over replications). Holm correction over "
              "all 9 tests:\n\n"
            + df_to_markdown(df_con, fmt) + "\n")
    if confirm is not None:
        cm = confirm["means"].rename(columns={"mean": "Mean", "se": "SE", "bias": "Bias"})
        text += (f"\nConfirmation run at S0 = {confirm['S0']:g}: R = {confirm['R']} replications, "
                 f"independent seed {confirm['seed']}, same design. Four pre-specified paired "
                 f"tests, Holm correction over the four:\n\n"
                 + df_to_markdown(cm, fmt) + "\n\n"
                 + df_to_markdown(confirm["table"], fmt) + "\n")
    write_result("benchmark_2x2", text)
    df_sum.to_csv("output_tables/benchmark_2x2.csv", index=False)
    return dict(summary=df_sum, vs_best=df_best, contrasts=df_con)


def confirm_2x2_large_R(S0: float = 100.0, R: int = 200, n_paths: int = 4000,
                       seed: int = 2027) -> dict:
    """
    Confirmation run of the 2x2 design at one S0 with more replications and an independent
    seed (CRN within each replication). Four pre-specified paired tests, Holm-corrected:
    TVR - LSM, target effect, sample effect, interaction (definitions as in analyze_2x2).
    """
    methods = tuple(METHOD_LABEL)
    pr = run_benchmark_replications(methods, (S0,), R=R, n_paths=n_paths, seed=seed)
    cf_itm, cf_all = pr[(S0, "LSM")], pr[(S0, "LSM-all")]
    v_itm, v_all = pr[(S0, "TVR-ITM")], pr[(S0, "TVR")]
    zero = np.zeros(R)
    tests = {
        "TVR - classic LSM": paired_t(v_all, cf_itm),
        "Target (value - cash flow)": paired_t(((v_itm - cf_itm) + (v_all - cf_all)) / 2, zero),
        "Sample (all - ITM)": paired_t(((cf_all - cf_itm) + (v_all - v_itm)) / 2, zero),
        "Interaction": paired_t((v_all - v_itm) - (cf_all - cf_itm), zero),
    }
    p_holm = holm([t["p"] for t in tests.values()])
    rows = [{"Contrast": k, "Estimate": t["diff"], "SE": t["se"], "t": t["t"], "p": t["p"],
             "p (Holm)": ph, "95% CI": f"[{t['ci_low']:.4f}, {t['ci_high']:.4f}]"}
            for (k, t), ph in zip(tests.items(), p_holm)]
    df = pd.DataFrame(rows)
    means = pd.DataFrame([{"Method": m, **{k: v for k, v in summarize(pr[(S0, m)], REFERENCE[S0]).items()
                                            if k in ("mean", "se", "bias")}} for m in methods])
    print(f"\nConfirmation run (S0 = {S0}, R = {R}, seed {seed})")
    print(means.to_string(index=False))
    print(df.to_string(index=False))
    return dict(prices=pr, tests=tests, table=df, means=means, S0=S0, R=R, seed=seed)

STULZ_CHECK_SETS = [
    # (label, S1, S2, K, r, T, q1, q2, sigma1, sigma2, rho)
    ("Glasserman benchmark, rho = 0", 100, 100, 100, 0.05, 3.0, 0.10, 0.10, 0.20, 0.20, 0.0),
    ("rho = 0.5, unequal sigmas", 100, 100, 100, 0.05, 1.0, 0.00, 0.00, 0.30, 0.20, 0.5),
    ("rho = -0.5, unequal S0 and q", 90, 110, 100, 0.03, 2.0, 0.02, 0.05, 0.25, 0.35, -0.5),
    ("rho = 0.8, high vol, OTM", 100, 95, 105, 0.05, 0.5, 0.00, 0.03, 0.60, 0.80, 0.8),
    ("rho = 0.9, ITM, sigma 0.15 / 0.40", 120, 110, 100, 0.04, 1.5, 0.01, 0.00, 0.15, 0.40, 0.9),
    ("rho = 0.3, deep OTM", 80, 85, 120, 0.05, 1.0, 0.00, 0.00, 0.20, 0.30, 0.3),
    ("rho = -0.9, short maturity", 100, 100, 100, 0.05, 0.25, 0.00, 0.00, 0.20, 0.20, -0.9),
]


def validate_stulz(n_paths: int = 2_000_000, seed: int = 2026) -> pd.DataFrame:
    """
    Compares the Stulz (1982) closed form with a direct Monte Carlo price of the European
    max-call (exact terminal sampling, unbiased) for several parameter sets.
    Check: |closed form - MC| <= 3 MC standard errors for every set.
    """
    print(f"\n--- Stulz closed form vs Monte Carlo (N = {n_paths:,}) ---")
    rows = []
    for j, (label, S1, S2, K, r, T, q1, q2, s1, s2, rho) in enumerate(STULZ_CHECK_SETS):
        cf = stulz_max_call(S1, S2, K, r, T, q1, q2, s1, s2, rho)
        mc, se = mc_max_call(S1, S2, K, r, T, q1, q2, s1, s2, rho, n_paths,
                             np.random.default_rng([seed, 3, j]))
        rows.append({"Case": label, "S1": S1, "S2": S2, "K": K, "r": r, "T": T, "q1": q1,
                     "q2": q2, "sigma1": s1, "sigma2": s2, "rho": rho, "Closed form": cf,
                     "MC": mc, "MC SE": se, "(MC - CF) / SE": (mc - cf) / se})
    df = pd.DataFrame(rows)
    print(df[["Case", "Closed form", "MC", "MC SE", "(MC - CF) / SE"]].to_string(index=False))
    ok = bool(np.all(np.abs(df["(MC - CF) / SE"]) <= 3.0))
    print(f"Check |MC - CF| <= 3 SE for all sets: {'PASS' if ok else 'FAIL'}")
    fmt = {"S1": "{:g}", "S2": "{:g}", "K": "{:g}", "r": "{:g}", "T": "{:g}", "q1": "{:g}",
           "q2": "{:g}", "sigma1": "{:g}", "sigma2": "{:g}", "rho": "{:g}",
           "Closed form": "{:.4f}", "MC": "{:.4f}", "MC SE": "{:.4f}", "(MC - CF) / SE": "{:+.2f}"}
    text = (f"# European max-call: Stulz closed form vs Monte Carlo\n\n"
            f"Direct Monte Carlo of the discounted European payoff with exact terminal sampling, "
            f"N = {n_paths:,} paths per case (generator default_rng([{seed}, 3, case])). "
            f"Generated by `theory_engine.py`. Check |MC - closed form| <= 3 SE: "
            f"{'passed for all cases' if ok else 'FAILED'}.\n\n"
            + df_to_markdown(df, fmt) + "\n")
    write_result("stulz_check", text)
    return df


def run_control_variate(method: str = "TVR", S0_values: tuple[float, ...] = (90.0, 100.0, 110.0),
                        R: int = 50, n_paths: int = 4000, seed: int = 2026) -> pd.DataFrame:
    """
    European max-call (Stulz closed form, dividend yield delta) as control variate in the
    forward pass, on the same replications as the 2x2 design (seed [seed, 2, k]), so the plain
    prices coincide with the 2x2 table. The CV coefficient is estimated on the training paths,
    so the CV estimator is unbiased for the policy value and stays low-biased.
    Reports per S0: European price, mean plain / CV price (SE over replications), bias and
    RMSE against the reference, mean pricing-path SE, the within-replication variance
    reduction factor Var(Y) / Var(Y - bX) on the pricing paths, and the ratio of the variances
    of the R price estimates (which also contains the training-path noise of the policy).
    """
    print(f"\n--- European control variate ({method}, R = {R}, N = {n_paths}) ---")
    b = BENCH
    rows = []
    for S0 in S0_values:
        euro = stulz_max_call(S0, S0, b["K"], b["r"], b["T"], b["delta"], b["delta"],
                              b["sigma"], b["sigma"], 0.0)
        res = []
        for k in range(R):
            rng = np.random.default_rng([seed, 2, k])
            t1, t2, p1, p2 = benchmark_paths(S0, n_paths, n_paths, rng)
            model = lsm_train_max(t1, t2, b["K"], b["r"], b["T"], method=method)
            res.append(lsm_price_max_cv(model, t1, t2, p1, p2, euro))
        plain = np.array([x["price_plain"] for x in res])
        cv = np.array([x["price"] for x in res])
        sp, sc = summarize(plain, REFERENCE[S0]), summarize(cv, REFERENCE[S0])
        rows.append({"S0": S0, "European (Stulz)": euro, "Reference": REFERENCE[S0],
                     "Mean plain": sp["mean"], "SE plain": sp["se"],
                     "Mean CV": sc["mean"], "SE CV": sc["se"],
                     "Bias CV": sc["bias"], "RMSE plain": sp["rmse"], "RMSE CV": sc["rmse"],
                     "Pricing SE plain": float(np.mean([x["se_plain"] for x in res])),
                     "Pricing SE CV": float(np.mean([x["se"] for x in res])),
                     "Corr(Y, X)": float(np.mean([x["corr"] for x in res])),
                     "VRF per run": float(np.mean([x["vrf"] for x in res])),
                     "Var ratio across runs": sp["var"] / sc["var"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    fmt = {"S0": "{:.0f}", "European (Stulz)": "{:.4f}", "Reference": "{:.2f}",
           "Mean plain": "{:.4f}", "SE plain": "{:.4f}", "Mean CV": "{:.4f}", "SE CV": "{:.4f}",
           "Bias CV": "{:+.4f}", "RMSE plain": "{:.4f}", "RMSE CV": "{:.4f}",
           "Pricing SE plain": "{:.4f}", "Pricing SE CV": "{:.4f}", "Corr(Y, X)": "{:.3f}",
           "VRF per run": "{:.2f}", "Var ratio across runs": "{:.2f}"}
    text = (f"# European max-call as control variate (Glasserman benchmark, {method} policy)\n\n"
            f"R = {R} replications per S0 (same paths as the 2x2 design, seed {seed}), "
            f"N = {n_paths:,} training and {n_paths:,} pricing paths. Control: discounted "
            f"European payoff with known mean (Stulz closed form, dividend yield "
            f"{b['delta']}); coefficient b estimated on the training paths. 'VRF per run' = "
            f"Var(Y) / Var(Y - bX) on the pricing paths, averaged over runs (pricing noise "
            f"only). 'Var ratio across runs' = variance of the R plain prices / variance of the "
            f"R CV prices (includes the training noise of the policy, which the control variate "
            f"cannot remove). Generated by `theory_engine.py`.\n\n"
            + df_to_markdown(df, fmt) + "\n")
    write_result("control_variate", text)
    return df

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
    text = ("# Glasserman Example 8.6.1: single run per S0 (TVR policy)\n\n"
            f"K = {K:g}, T = {T:g}, r = {r}, delta = {delta}, sigma = {sigma}, rho = 0, "
            f"{m_steps} exercise dates, N = {n_paths:,} training and {n_paths:,} independent "
            f"pricing paths, generator default_rng([{seed}, S0]), basis degree {DEFAULT_DEGREE} "
            f"scaled by K. Std. error = pricing-path standard error conditional on the trained "
            f"policy. For 50 replications per S0 and all four methods see "
            f"[benchmark_2x2.md](benchmark_2x2.md). Generated by `theory_engine.py`.\n\n"
            + df_to_markdown(df_results.rename(columns={
                "LSM_Price": "Price", "Std_Error": "Std. error", "Exact_Price": "Reference",
                "Difference": "Difference", "Diff_over_SE": "Difference / SE"}),
                {"S0": "{:.0f}", "Price": "{:.3f}", "Std. error": "{:.3f}", "Reference": "{:.2f}",
                 "Difference": "{:+.3f}", "Difference / SE": "{:+.2f}"}))
    write_result("glasserman_replication", text)
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

def compare_lsm_tvr_theory(lsm_prices: np.ndarray, tvr_prices: np.ndarray,
                           S0: float = 100.0, n_paths: int = 4000, seed: int = 2026,
                           supplementary: dict | None = None) -> pd.DataFrame:
    """
    Variance and mean comparison of classic LSM and TVR at one S0, on the replicated
    benchmark prices (both methods on the same paths in each replication, so the samples are
    paired). Reports mean, variance and 95% CI per method, the F-test (assumes independence)
    and the Pitman-Morgan test (paired) for equal variances, and the paired t-test for the
    mean difference TVR - LSM. Both prices are low-biased, so a positive mean difference
    indicates a better TVR policy.
    """
    print(f"\n--- Classic LSM vs TVR at S0 = {S0} ({len(lsm_prices)} replications) ---")
    x, y = np.asarray(lsm_prices), np.asarray(tvr_prices)
    num_runs = len(x)
    rows = []
    for name, v in (("Classic LSM", x), ("TVR", y)):
        st = summarize(v, REFERENCE[S0])
        rows.append({"Method": name, "Mean": st["mean"], "SE": st["se"], "Variance": st["var"],
                     "CI 95% low": st["ci_low"], "CI 95% high": st["ci_high"]})
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    f_stat = np.var(x, ddof=1) / np.var(y, ddof=1)
    df1 = df2 = num_runs - 1
    f_p = 2 * min(stats.f.cdf(f_stat, df1, df2), stats.f.sf(f_stat, df1, df2))
    pm = pitman_morgan(x, y)
    pair_corr = float(np.corrcoef(x, y)[0, 1])
    t = paired_t(y, x)
    print(f"F-test: F = {f_stat:.4f}, p = {f_p:.4f}; correlation {pair_corr:.4f}; "
          f"Pitman-Morgan r = {pm['r']:.4f}, p = {pm['p']:.4f}")
    print(f"Paired t-test TVR - LSM: {t['diff']:.4f} (SE {t['se']:.4f}, t = {t['t']:.4f}, "
          f"p = {t['p']:.4g}, 95% CI [{t['ci_low']:.4f}, {t['ci_high']:.4f}])")

    fmt = {"Mean": "{:.4f}", "SE": "{:.4f}", "Variance": "{:.4f}", "CI 95% low": "{:.4f}",
           "CI 95% high": "{:.4f}"}
    text = (f"# Classic LSM vs TVR (S0 = {S0:g}, {num_runs} replications)\n\n"
            f"Same replications as the 2x2 design ([benchmark_2x2.md](benchmark_2x2.md)): "
            f"N = {n_paths:,} training and pricing paths, both methods on the same paths in each "
            f"replication (seed {seed}). Generated by `theory_engine.py`.\n\n"
            + df_to_markdown(df_summary, fmt)
            + "\n\nTests for equal variances (H0: Var(LSM) = Var(TVR)):\n\n"
            + "| Test | Statistic | p-value |\n|---|---:|---:|\n"
            + f"| F-test, two-sided, df = ({df1}, {df2}) | F = {f_stat:.4f} | {f_p:.4f} |\n"
            + f"| Pitman-Morgan (paired samples) | r = {pm['r']:.4f} | {pm['p']:.4f} |\n\n"
            + f"Correlation of LSM and TVR prices across replications: {pair_corr:.4f}. The "
              f"F-test assumes independent samples; Pitman-Morgan accounts for the pairing.\n\n"
            + f"Paired t-test for the mean difference (H0: E[TVR - LSM] = 0), df = {num_runs - 1}:"
              f"\n\n| Mean difference TVR - LSM | SE | t | p-value | 95% CI |\n"
              f"|---:|---:|---:|---:|---|\n"
              f"| {t['diff']:+.4f} | {t['se']:.4f} | {t['t']:.4f} | {fmt_p(t['p'])} | "
              f"[{t['ci_low']:.4f}, {t['ci_high']:.4f}] |\n")
    if supplementary is not None:
        ts = supplementary["test"]
        text += (f"\nSupplementary confirmation run (S0 = {supplementary['S0']:g}, "
                 f"R = {supplementary['R']}, independent seed {supplementary['seed']}, same "
                 f"design): TVR - LSM = {ts['diff']:+.4f} (SE {ts['se']:.4f}, t = {ts['t']:.2f}, "
                 f"p = {fmt_p(ts['p'])}, 95% CI [{ts['ci_low']:.4f}, {ts['ci_high']:.4f}]).\n")
    write_result("lsm_vs_tvr", text)
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

    # 3.0 Replicated benchmark at S0 = 90/100/110: 2x2 design (target x sample)
    methods = tuple(METHOD_LABEL)
    bench = run_benchmark_replications(methods, R=50, n_paths=n_paths)
    confirm = confirm_2x2_large_R(S0=S0_base, R=200, n_paths=n_paths)
    analyze_2x2(bench, R=50, n_paths=n_paths, confirm=confirm)

    # 3.0b Classic LSM vs TVR at S0 = 100 on the same replications (variance tests, paired t)
    supp = dict(test=confirm["tests"]["TVR - classic LSM"], S0=confirm["S0"], R=confirm["R"],
                seed=confirm["seed"])
    compare_lsm_tvr_theory(bench[(S0_base, "LSM")], bench[(S0_base, "TVR")],
                           S0=S0_base, n_paths=n_paths, supplementary=supp)

    # 4.0 European closed form (Stulz) vs Monte Carlo, and control variate
    validate_stulz()
    run_control_variate("TVR", R=50, n_paths=n_paths)

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