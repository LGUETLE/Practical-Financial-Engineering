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

- Removed an archived prototype script from the repository and rewrote `.gitignore`, so that
  generated data, plots and tables stay out of version control.
- `empirical_engine.py`: the risk-neutral drift `r` is now passed explicitly to the simulation
  (the historically estimated drift is not used for pricing); added a 2D volatility sensitivity
  plot comparing LSM and TVR.
- Added `requirements.txt` pinned to the tested package versions.
- `lsm_price_max` can return the standard error of the price estimate (`return_se=True`).
- The LSM vs TVR comparison uses a fixed seed, so its results are reproducible.
- Corrected the docstring of the regression basis: it is a complete degree-2 polynomial plus the
  intrinsic value (7 terms), not a degree-3 basis.
- `theory_engine.py` now reports standard errors and the difference to the reference values,
  tests the LSM/TVR variance difference (F-test and Pitman–Morgan test), and exports the
  exercise-region plot as PNG.
- Added this README and the `results/` folder with tables and figures from a documented run.

## Model and methods

**Model.** Under the risk-neutral measure the two assets follow independent geometric Brownian
motions with dividend yield δ,
dS_j(t) = (r − δ) S_j(t) dt + σ S_j(t) dW_j(t), j = 1, 2. The option can be exercised at
m = 9 equally spaced dates t_i = iT/9 up to maturity and pays h(S_1, S_2) = max(max(S_1, S_2) − K, 0).
Benchmark parameters (Glasserman, Example 8.6.1): K = 100, T = 3, r = 0.05, δ = 0.10, σ = 0.20,
S_0 ∈ {90, 100, 110}. Paths are simulated exactly on the exercise grid.

**Regression basis.** At each exercise date the continuation value is regressed on 7 functions:
1, S_1, S_2, S_1², S_2², S_1·S_2 and the intrinsic value h(S_1, S_2).

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
price is the mean of the discounted payoffs at the chosen stopping times. Because any stopping rule
is at most optimal, both estimators are low-biased (E[V̂] ≤ true price). LSM and TVR differ only in
the exercise policy learned in the backward pass; a higher mean price indicates a better policy.

## Results

All numbers below come from one run of `theory_engine.py` (Python 3.13, package versions as in
`requirements.txt`). The tables are also in [`results/`](results/).

### Replication of Glasserman Example 8.6.1 (TVR policy)

N = 4,000 training paths and 4,000 independent pricing paths per S_0. The standard error is that
of the pricing paths, conditional on the trained policy.

| S0 | Price | Std. error | Reference | Difference | Difference / SE |
|---:|---:|---:|---:|---:|---:|
| 90 | 7.990 | 0.193 | 8.08 | -0.090 | -0.47 |
| 100 | 13.760 | 0.242 | 13.90 | -0.140 | -0.58 |
| 110 | 21.552 | 0.286 | 21.34 | +0.212 | +0.74 |

All three prices lie within one standard error of the reference values, so they are consistent
with them. Single runs with standard errors of 0.19–0.29 cannot resolve a low bias of order 0.1;
the S_0 = 110 estimate above the reference is within the Monte Carlo noise.

### Classic LSM vs TVR (S0 = 100, 50 replications)

Each replication uses new training and pricing paths (N = 4,000 each); LSM and TVR are evaluated on
the same paths.

| Method | Mean price | Variance | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|---:|
| Classic LSM | 13.7766 | 0.0555 | 13.7097 | 13.8436 |
| TVR | 13.8060 | 0.0505 | 13.7422 | 13.8699 |

| Test (H0: equal variances) | Statistic | p-value |
|---|---:|---:|
| F-test, two-sided, df = (49, 49) | F = 1.0991 | 0.7421 |
| Pitman–Morgan (paired samples) | r = 0.1879 | 0.1914 |

- **Variance.** The variance difference between LSM and TVR is not statistically significant
  (F-test p = 0.74). Because both methods are priced on the same paths, their prices are strongly
  correlated across replications (correlation 0.969), which violates the independence assumption of
  the F-test; the Pitman–Morgan test for paired samples does not reject equal variances either
  (p = 0.19).
- **Bias.** Both 95% confidence intervals lie entirely below the reference value 13.90 (by about
  0.12 for LSM and 0.09 for TVR in the mean). This is consistent with the expected low bias of a
  price computed with an estimated, suboptimal exercise policy.
- **LSM vs TVR mean.** TVR has the higher mean price, but the confidence intervals overlap
  substantially and the mean difference has not been tested here, so these results do not show
  that one policy is better.

### Estimated exercise region

Exercise region at exercise date i = 4 (t = 4/3), from a TVR model trained on 20,000 paths. Exercise
is optimal when one asset is well above the strike and clearly above the other; near the diagonal
S_1 = S_2 it is optimal to continue.

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
