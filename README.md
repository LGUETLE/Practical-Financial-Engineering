# Bermudan Max-Call Pricing with Regression-Based Monte Carlo

Monte Carlo pricing of a Bermudan call on the maximum of two assets with regression-based methods.
The project replicates Example 8.6.1 of Glasserman (2003) and compares two regression targets for
the continuation value: classic Longstaff–Schwartz (LSM) and Tsitsiklis–Van Roy (TVR). Both
methods learn an exercise policy on training paths and are then priced on independent paths, so
every reported price is a low-biased estimate with a standard error or confidence interval.

## Background and changes

Originally developed as a group project (three students) in the Project in Business Mathematics,
Institute of Financial Mathematics, University of Ulm, July 2026. I led the project and wrote the
code. Since then I have revised and extended it individually:

- Standard errors: `lsm_price_max` can return the standard error of the price estimate
  (`return_se=True`); the Glasserman replication now reports it together with the difference to
  the reference values.
- Statistical tests for the LSM vs TVR comparison: F-test and Pitman–Morgan test for the
  variances, paired t-test for the mean difference.
- The LSM vs TVR comparison uses a fixed seed, so its results are reproducible.
- Corrected the description of the regression basis: it is a complete degree-2 polynomial plus
  the intrinsic value (7 terms), not a degree-3 basis.
- `empirical_engine.py`: the risk-neutral drift `r` is now passed explicitly to the simulation
  (the historically estimated drift is not used for pricing); added a 2D volatility sensitivity
  plot comparing LSM and TVR.
- `results/` folder with the tables and figures shown below, generated from a documented run.
- Reproducibility: all randomness comes from `np.random.default_rng(seed)` generators passed
  explicitly to the simulation functions (no global NumPy seed). This changed the random stream:
  on the new stream the paired TVR − LSM difference at S0 = 100 is −0.0039 (SE 0.0097, p = 0.69),
  so the earlier significant result (p = 0.0008) did not replicate (see the 2x2 comparison below).
- Regression basis: degree 2 or complete degree 3 (+ intrinsic value), evaluated on S/K. A basis
  ablation chose degree 2 scaled by K as default.
- 2x2 design separating the regression target (realized cash flow vs value iteration) from the
  regression sample (ITM vs all paths), with paired tests and Holm correction.
- European max-call closed form (Stulz 1982), validated against Monte Carlo, and used as a control
  variate in the forward pass.
- Upper bound by duality (Andersen–Broadie 2004) with nested simulation, giving a price interval
  for every learned policy (`duality.py`, `bounds_engine.py`).

## Model and methods

**Model.** Under the risk-neutral measure the two assets follow independent geometric Brownian
motions with dividend yield δ,
dS_j(t) = (r − δ) S_j(t) dt + σ S_j(t) dW_j(t), j = 1, 2. The option can be exercised at
m = 9 equally spaced dates t_i = iT/9 up to maturity and pays h(S_1, S_2) = max(max(S_1, S_2) − K, 0).
Benchmark parameters (Glasserman, Example 8.6.1): K = 100, T = 3, r = 0.05, δ = 0.10, σ = 0.20,
S_0 ∈ {90, 100, 110}. Paths are simulated exactly on the exercise grid.

**Regression basis.** At each exercise date the continuation value is regressed on a complete
polynomial in x_j = S_j / K plus the intrinsic value h / K. Degree 2 (default): 1, x_1, x_2, x_1²,
x_2², x_1·x_2, h/K (7 terms); degree 3 adds x_1³, x_1²x_2, x_1x_2², x_2³ (11 terms). Dividing by
the fixed strike K uses no information from the simulated paths; it is a constant rescaling of the
columns, so it leaves the fitted policy unchanged and only improves the conditioning.

**Basis ablation** (S0 = 100, R = 50 replications, N = 4,000, all configurations on the same paths;
full table in [`results/basis_ablation.md`](results/basis_ablation.md)):

| Method | Basis | Mean (SE) | median log10 cond, raw → S/K |
|---|---|---|---|
| Classic LSM | degree 2 | 13.8117 (0.0337) | 6.0 → 2.3 |
| Classic LSM | degree 3 | 13.7831 (0.0319) | 8.7 → 3.3 |
| TVR | degree 2 | 13.8340 (0.0311) | 5.6 → 2.1 |
| TVR | degree 3 | 13.8218 (0.0305) | 8.3 → 3.0 |

Scaling lowers the condition number by 3–5 orders of magnitude but gives bit-identical prices on
all 50 runs: in double precision the raw basis is not ill-conditioned enough to change a single
exercise decision. Degree 3 gives a *lower* mean price than degree 2 (paired difference −0.0286,
Holm-adjusted p = 0.011 for LSM; −0.0123, p = 0.14 for TVR): with 4,000 training paths the four
extra coefficients add more estimation noise to the policy than they remove approximation error.
Since both estimators are low-biased, degree 2 (scaled) is the default.

**Longstaff–Schwartz (LSM).** The backward pass regresses the *realized* discounted cash flow of
each path (the payoff obtained by following the policy estimated at later dates) on the basis
functions, using only in-the-money paths. The fitted value is used only for the exercise decision:
a path is exercised if the intrinsic value exceeds the fitted continuation value; otherwise it keeps
its realized future cash flow. Regression errors therefore affect the decision but do not enter
the propagated values.

**Tsitsiklis–Van Roy (TVR).** The backward pass regresses on *all* paths, and the regression target
at date t_i is the discounted value estimate from date t_{i+1}, max(h_{i+1}, Ĉ_{i+1}), i.e. a
regression-based value iteration. The fitted continuation values themselves are propagated
backwards, so regression errors can accumulate across dates.

**Pricing.** Both methods are priced with the same out-of-sample forward pass: the learned
coefficients define a stopping rule that is applied to an independent set of pricing paths, and the
price is the mean of the discounted payoffs at the chosen stopping times. Because no stopping rule
can outperform the optimal one, both estimators are low-biased (E[V̂] ≤ true price). LSM and TVR differ only in
the exercise policy learned in the backward pass; a higher mean price indicates a better policy.

## Results

All numbers below come from one run of `theory_engine.py` (Python 3.13, package versions as in
`requirements.txt`). The tables are also in [`results/`](results/).

### Glasserman benchmark: 2x2 design (target x sample), 50 replications per S0

Four regression methods: target {realized cash flow, value iteration} x sample {ITM paths,
all paths}; classic LSM = (cash flow, ITM), TVR = (value, all). R = 50 replications per S0,
N = 4,000 training and 4,000 pricing paths; in each replication all methods use the same paths
(common random numbers). Full tables: [`results/benchmark_2x2.md`](results/benchmark_2x2.md).

| S0 | Reference | Classic LSM | Cash flow, all | Value, ITM | TVR |
|---:|---:|---:|---:|---:|---:|
| 90 | 8.08 | 8.0059 (0.0285) | 8.0242 (0.0274) | 8.0148 (0.0272) | 8.0201 (0.0289) |
| 100 | 13.90 | 13.8442 (0.0322) | 13.8729 (0.0321) | 13.8619 (0.0324) | 13.8684 (0.0331) |
| 110 | 21.34 | 21.2448 (0.0381) | 21.2542 (0.0381) | 21.2449 (0.0349) | 21.2464 (0.0374) |

Mean (SE over replications). All twelve means lie below the reference, as expected for low-biased
estimators. (Cash flow, all paths) has the highest mean at every S0; after a Holm correction over
the nine comparisons with it, only classic LSM at S0 = 100 is significantly worse (−0.029,
adjusted p = 0.006).

Confirmation run at S0 = 100 with R = 200 and an independent seed (four pre-specified paired
tests, Holm-corrected):

| Contrast | Estimate | SE | Holm p |
|---|---:|---:|---:|
| TVR − classic LSM | +0.0226 | 0.0052 | 6.7e-05 |
| Target (value − cash flow) | −0.0003 | 0.0041 | 0.94 |
| Sample (all − ITM) | +0.0229 | 0.0034 | 1.1e-09 |
| Interaction | +0.0049 | 0.0052 | 0.69 |

**The regression sample, not the regression target, drives the difference between LSM and TVR.**
Fitting on all paths instead of ITM paths only raises the price by about 0.023 (a better policy);
regressing on value-iteration targets instead of realized cash flows makes no detectable
difference. TVR − LSM equals the sum of the two main effects, so it is almost entirely the sample
effect. The effect is small (about 0.02 on 13.8, i.e. 0.15%).

### Classic LSM vs TVR (S0 = 100): variance tests

On the 50 replications above ([`results/lsm_vs_tvr.md`](results/lsm_vs_tvr.md)): variances
0.0517 (LSM) and 0.0549 (TVR), F-test p = 0.83, Pitman–Morgan (paired) p = 0.47, so no evidence of
a variance difference. Paired mean difference TVR − LSM = +0.0243 (SE 0.0094, p = 0.013).
The stage-1 run (old random stream) gave +0.0294 (p = 0.0008); the first run on the new stream
in this revision gave −0.0039 (p = 0.69). With R = 200 the difference is +0.0226 (SE 0.0052), so
the stage-1 direction holds, but single 50-run experiments of this size can miss it.

Single run per S0 (TVR, N = 4,000): see
[`results/glasserman_replication.md`](results/glasserman_replication.md); deviations from the
reference of up to 1.3 standard errors are Monte Carlo noise.

### European closed form and control variate

`european.py` implements the Stulz (1982) price of a European call on the maximum of two assets
with dividend yields and correlation. On seven parameter sets (ρ ∈ {0, ±0.5, 0.3, 0.8, 0.9, −0.9},
unequal volatilities, spot prices and dividend yields) it agrees with a direct Monte Carlo price
(N = 2,000,000) within 1.45 standard errors ([`results/stulz_check.md`](results/stulz_check.md)).

The discounted European payoff is used as a control variate for the TVR policy; its coefficient
is estimated on the training paths, so the estimator remains unbiased for the policy value and
hence low-biased ([`results/control_variate.md`](results/control_variate.md), same 50
replications as above):

| S0 | European (Stulz) | Mean plain (SE) | Mean CV (SE) | Corr(Y, X) | VRF per run | Var ratio across runs |
|---:|---:|---:|---:|---:|---:|---:|
| 90 | 6.6551 | 8.0201 (0.0289) | 8.0186 (0.0217) | 0.69 | 1.92 | 1.78 |
| 100 | 11.1957 | 13.8684 (0.0331) | 13.8650 (0.0266) | 0.65 | 1.71 | 1.56 |
| 110 | 16.9286 | 21.2464 (0.0374) | 21.2390 (0.0307) | 0.60 | 1.56 | 1.49 |

The variance reduction is modest (factor 1.5–1.9): with δ = 0.10 early exercise is worth a lot
(13.90 vs 11.20 at S0 = 100), so the Bermudan and European payoffs are only moderately correlated.

### Price bounds by duality (Andersen–Broadie)

For a learned policy, the lower bound L is its out-of-sample value (200,000 paths). The upper bound
U = L + gap comes from the Andersen–Broadie martingale built from the policy's own continuation
values, which are estimated by nested simulation (4,000 outer paths, 2,000 inner paths per
in-the-money date). U is high-biased for any number of inner paths. Full table:
[`results/duality_bounds.md`](results/duality_bounds.md).

| S0 | Policy | L (SE) | U (SE) | Gap (SE) | 95% interval | Reference |
|---:|---|---:|---:|---:|---|---:|
| 90 | TVR, 4,000 training paths (default) | 7.9972 (0.0278) | 8.0679 (0.0282) | 0.0707 (0.0047) | [7.9427, 8.1232] | 8.08 |
| 90 | TVR, 100,000 training paths | 8.0111 (0.0276) | 8.0552 (0.0278) | 0.0441 (0.0034) | [7.9571, 8.1097] | 8.08 |
| 100 | TVR, 4,000 training paths (default) | 13.8472 (0.0342) | 13.9869 (0.0354) | 0.1397 (0.0089) | [13.7801, 14.0563] | 13.90 |
| 100 | TVR, 100,000 training paths | 13.8743 (0.0350) | 13.9549 (0.0355) | 0.0806 (0.0058) | [13.8056, 14.0245] | 13.90 |
| 110 | TVR, 4,000 training paths (default) | 21.2174 (0.0405) | 21.4043 (0.0415) | 0.1868 (0.0090) | [21.1381, 21.4855] | 21.34 |
| 110 | TVR, 100,000 training paths | 21.2530 (0.0406) | 21.3704 (0.0411) | 0.1174 (0.0067) | [21.1735, 21.4510] | 21.34 |

The Glasserman reference lies inside the 95% interval in all 12 cases (four policies per S0).
Policies evaluated on the same paths rank identically by lower bound and by gap: a better policy
(higher L) has a smaller gap. At S0 = 100 the gaps are 0.081 (TVR, 100,000 paths) < 0.140 (TVR,
4,000) < 0.161 (classic LSM, 4,000) < 0.442 (TVR, 500). The gap separates policies more sharply
than L: at S0 = 110 classic LSM and TVR have indistinguishable lower bounds (difference −0.0095,
SE 0.0118), but the LSM gap is larger by 0.025 (SE 0.008, Holm p = 0.002). Runtime about 70 s
for all 12 bound pairs.

### Estimated exercise region

Estimated exercise region at exercise date i = 4 (t = 4/3), from a TVR model trained on 20,000
paths. The estimated policy exercises when one asset is well above the strike and clearly above the
other, and continues near the diagonal S_1 = S_2.

![Estimated exercise region at step 4](results/exercise_boundary.png)

## Work in progress

The empirical calibration (`empirical_engine.py`, `data_fetcher.py`: volatilities and correlation
estimated from AAPL/MSFT and BTC/ETH prices) and the stress test (`stresstest_engine.py`) are being
revised and are not reported here yet. Planned next: the European max-call closed form (Stulz 1982)
as benchmark and control variate, a duality-based upper bound (Andersen–Broadie 2004), and `pytest`
tests.

## Project structure

```
README.md                       this file
requirements.txt                pinned dependencies
american_options_functions.py   GBM simulation (independent + correlated), basis, LSM/TVR train + price
theory_engine.py                Glasserman benchmark: LSM vs TVR, moneyness, volatility, convergence, exercise region
empirical_engine.py             calibration on AAPL/MSFT and BTC/ETH, macro-replications, 2D sensitivity (in revision)
stresstest_engine.py            estimator variance at low path counts and high volatility (in revision)
data_fetcher.py                 historical prices, estimation of sigma and rho, cache in data/
results/                        curated figures and tables shown in this README
```

Generated (not committed): `data/`, `output_plots/`, `output_tables/`.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python theory_engine.py
```

`theory_engine.py` runs in well under a minute and writes LaTeX/CSV tables to `output_tables/`
and plots to `output_plots/`. `empirical_engine.py` downloads historical prices and needs network
access.

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer, Ch. 8.
- Longstaff, F. A., Schwartz, E. S. (2001). Valuing American options by simulation: a simple
  least-squares approach. *Review of Financial Studies* 14(1), 113–147.
- Tsitsiklis, J. N., Van Roy, B. (2001). Regression methods for pricing complex American-style
  options. *IEEE Transactions on Neural Networks* 12(4), 694–703.
- Stulz, R. (1982). Options on the minimum or the maximum of two risky assets. *Journal of
  Financial Economics* 10(2), 161–185.
- Andersen, L., Broadie, M. (2004). Primal-dual simulation algorithm for pricing multidimensional
  American options. *Management Science* 50(9), 1222–1234.
