# Changelog

Originally developed as a group project (three students) in the Project in Business Mathematics,
Institute of Financial Mathematics, University of Ulm, July 2026. I led the project and wrote the
code. Since then I have revised and extended it individually.

## Stages 2 and 3 (November 2026)

Changes compared with the course version:

- **Statistics.** Every price comes with a standard error or confidence interval. Methods are
  compared on common random numbers with paired tests, and a Holm correction is applied when
  several comparisons are tested together.
- **Reproducibility.** All randomness comes from `np.random.default_rng(seed)` generators passed
  explicitly to the simulation functions (no global NumPy seed).
- **Stage-1 LSM vs TVR result reconciled.** Stage 1 reported TVR − LSM = +0.029 (p = 0.0008)
  from one 50-replication run with `np.random.seed(2026)`. The same experiment with
  `np.random.default_rng(2026)` gave −0.004 (p = 0.69). Only the random stream differed:
  - Code: the current code fed the legacy stream reproduces the stage-1 prices to 1e-15, and the
    stage-1 and current code give identical prices on identical paths.
  - Seed study: across 80 runs of the same configuration (40 seeds per generator), the
    difference is +0.018 on average (SE 0.001), with an SD of 0.010 for a single 50-run
    estimate. The two generators do not differ (p = 0.37).
  - Verdict: stage 1 was a somewhat high draw (+1.2 SD) and the rerun an unusually low one
    (−2.3 SD; none of the 80 study runs was lower). Its direction holds: TVR is slightly better.
    But a single 50-run experiment detects the effect at the 5% level in only 34 of 80 runs, and
    the stage-1 estimate overstated its size. See [results.md](results.md#seed-study-stage-1-vs-rerun).
- **Regression basis.** Degree 2 (7 terms) or complete degree 3 (11 terms) plus the intrinsic value,
  evaluated on S/K. An ablation chose degree 2, scaled, as default.
- **2x2 design.** It separates the regression target from the regression sample; the two methods
  not covered by classic LSM and TVR were added.
- **European closed form.** The Stulz (1982) price of the European max-call, validated against
  Monte Carlo and used as a control variate.
- **Upper bound by duality.** Andersen–Broadie (2004) with nested simulation, giving a price
  interval for any learned policy.
- **Empirical part rebuilt.**
  - Data: Yahoo Finance for stocks and crypto (Binance removed), one fixed window
    2022-01-01 to 2025-12-31.
  - Calibration: only σ and ρ (the unused drift estimate was removed).
  - Reference: the exact European price, since with δ = 0 early exercise is never optimal.
  - The script no longer crashes on current matplotlib.
- **Stress test rebuilt.** Seeded, with all four methods, variance tests and a reference interval.
- **Engineering.** Package structure, a single `run_all.py` that regenerates `results/`, a
  `pytest` suite and a GitHub Actions workflow.
- **Documentation.** The README is a short overview; methods, results and this changelog are in
  `docs/`.

## Stage 1 (after the course)

- Standard errors: `lsm_price_max` can return the standard error of the price estimate.
- Statistical tests for the LSM vs TVR comparison: F-test and Pitman–Morgan test for the
  variances, paired t-test for the mean difference.
- Fixed seed for the LSM vs TVR comparison (legacy global `np.random.seed`).
- Corrected the description of the regression basis: a complete degree-2 polynomial plus the
  intrinsic value (7 terms), not a degree-3 basis.
- `empirical_engine.py`: the risk-neutral drift `r` passed explicitly to the simulation; 2D
  volatility sensitivity plot comparing LSM and TVR.
- `results/` folder with tables and figures from a documented run.

