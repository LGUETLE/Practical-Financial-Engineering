# Methods

Model, estimators, bounds and statistics used in this project. Results are in
[results.md](results.md), the change history in [changelog.md](changelog.md).

## Overview

Monte Carlo pricing of a Bermudan call on the maximum of two assets with regression-based methods.
The project replicates Example 8.6.1 of Glasserman (2003) and compares four ways of estimating the
continuation value. Classic Longstaff–Schwartz (LSM) and Tsitsiklis–Van Roy (TVR) are two corners
of a 2x2 design: regression target (realized cash flow vs value iteration) x regression sample
(in-the-money paths vs all paths). Every learned exercise policy is priced out of sample, which
gives a low-biased estimate. A duality-based upper bound (Andersen–Broadie) turns each estimate into
a price interval. The European closed form (Stulz 1982) serves as a control variate and as an exact
reference in calibrated scenarios.

## Model and estimators

**Model.** Under the risk-neutral measure each asset follows a geometric Brownian motion with
dividend yield δ, dS_j = (r − δ) S_j dt + σ_j S_j dW_j, with correlation ρ between the Brownian
motions. The option can be exercised at m equally spaced dates t_i = iT/m. It pays
h = max(max(S_1, S_2) − K, 0). Paths are simulated exactly on the exercise grid.

Benchmark (Glasserman, Example 8.6.1): K = 100, T = 3, r = 0.05, δ = 0.10, σ_1 = σ_2 = 0.20,
ρ = 0, m = 9, S0 ∈ {90, 100, 110}. Reference values 8.08 / 13.90 / 21.34 (binomial lattice).

**Regression basis.** At each exercise date the continuation value is regressed on a complete
polynomial in x_j = S_j / K plus the intrinsic value h / K.
- Degree 2 (default, 7 terms): 1, x_1, x_2, x_1², x_2², x_1x_2, h/K.
- Degree 3 (11 terms): adds x_1³, x_1²x_2, x_1x_2², x_2³.

Dividing by the fixed strike uses no information from the simulated paths. It is a constant
rescaling of the columns, which leaves the least-squares fit unchanged and only improves the
conditioning.

**Four regression methods (2x2 design).** In the backward pass, the coefficients at date t_i are
estimated by least squares.

| | regression on ITM paths | regression on all paths |
|---|---|---|
| **target: realized cash flow** | classic LSM | cash flow, all paths |
| **target: value iteration** | value iteration, ITM | TVR |

- *Realized cash flow* (Longstaff–Schwartz): the target is the discounted cash flow the path
  receives under the policy already estimated for later dates. The fit is used only for the
  exercise decision, and the path keeps its realized cash flow.
- *Value iteration* (Tsitsiklis–Van Roy): the target is the discounted value estimate
  max(h_{i+1}, Ĉ_{i+1}), and the fitted values themselves are propagated backwards.
- *ITM vs all paths*: the regression uses either only paths with h > 0 or all paths. For
  "value iteration, ITM", the continuation value of out-of-the-money paths is the fitted function
  extrapolated outside its fitting region.

**Pricing and bias direction.** All four methods are priced with the same forward pass. The learned
coefficients define a stopping rule: exercise at t_i if h_i > max(φ(S_i)ᵀβ_i, 0). The rule is
applied to independent pricing paths, and the price is the mean discounted payoff. No stopping rule
beats the optimal one, so every estimate is **low-biased** (E[V̂] ≤ true price). A higher mean
therefore means a better policy.

**European closed form.** The Stulz (1982) / Johnson (1987) formula prices the European call on
the maximum of two correlated assets with dividend yields. The bivariate normal CDF is computed by
adaptive quadrature (`maxcall/european.py`).

**Control variate.** The control is the discounted European payoff, whose mean is known from the
Stulz formula. Its coefficient is estimated on the training paths, so the estimator stays exactly
unbiased for the value of the policy, and thus low-biased for the Bermudan price.

**Upper bound by duality.** For the learned policy, build the Andersen–Broadie martingale
M_i = Σ_{k<i} (L_{k+1} − Q_k). Here L_k is the value of following the policy from t_k on, and
Q_k = E_k[L_{k+1}] is its continuation value, estimated by nested (inner) simulation (Q_0 is
estimated by the lower bound L). Then V_0 ≤ U = E[max_i (h_i − M_i)] = L + gap, with a pathwise
non-negative gap (Andersen and Broadie 2004; Glasserman 2003, Sec. 8.7):
- U stays high-biased for any number of inner paths, because the inner estimates enter M linearly
  and Jensen's inequality applies.
- The maximum is restricted to in-the-money dates and maturity. This is valid because stopping
  with h = 0 is never better than waiting.
- The implementation of the telescoped gap is checked against a brute-force martingale in the
  tests.

The reported interval is [L − 1.96 SE_L, U + 1.96 SE_U].

**Calibrated scenarios.** σ_1, σ_2 and ρ are estimated from daily log-returns.
- Historical returns are observed under P, and their drift is not used: pricing uses the
  risk-neutral drift r. Volatility and correlation are the same under P and Q.
- These scenarios have δ = 0. Then e^{−rt} max(S_1, S_2) is a submartingale under Q, so early
  exercise is never optimal and the Bermudan price equals the European price.
- The Stulz formula is therefore an exact reference for the bias of every method.

**Statistics.**
- Methods are compared on common random numbers: the same training and pricing paths in each
  replication.
- Comparisons use paired t-tests; variances are compared with the Pitman–Morgan test for paired
  samples.
- Holm's correction is applied whenever several hypotheses are tested together.
- Bias and RMSE are measured against the reference over replications.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                    # all 47 tests, about 2 s
pytest -m "not slow"      # 45 fast tests, run by GitHub Actions on every push
```

The tests cover:
- **GBM simulation:** the risk-neutral mean E[S_T] = S0 e^{(r−δ)T}, volatilities, and the
  correlation of the log-returns.
- **Stulz formula:** agreement with Monte Carlo, symmetry in swapping the assets, bounds by the
  single-asset calls, the large-strike limit, the Margrabe limit, and the bivariate normal CDF.
- **Basis:** shapes, column scaling, and invariance of the policy under scaling.
- **Reproducibility** of all four methods, and the **control variate**.
- **Duality bound:** the telescoped gap against a brute-force martingale, and L ≤ U.
- **Slow end-to-end checks:** the benchmark price at S0 = 100 lies within 3 SE of 13.90 (13.829,
  SE 0.050), and the reference lies inside the duality interval.

## Project structure

```
README.md  requirements.txt  requirements-dev.txt  pytest.ini
run_all.py              regenerates all results
theory_engine.py        Glasserman benchmark: ablation, 2x2 design, control variate, single runs
bounds_engine.py        lower and upper bounds (Andersen-Broadie duality)
empirical_engine.py     calibrated tech / crypto scenarios, bias against the Stulz price
stresstest_engine.py    low path counts at high volatility
seed_study.py           stage-1 LSM vs TVR comparison over many seeds and two generators
maxcall/
  simulation.py         exact GBM simulation (independent and correlated)
  lsm.py                basis, the four regression methods, forward pass, control variate
  european.py           Stulz closed form, Margrabe, Black-Scholes, direct Monte Carlo
  duality.py            Andersen-Broadie upper bound with nested simulation
  mc_stats.py           summaries, paired tests, Pitman-Morgan, Holm, Markdown tables
  data.py               Yahoo Finance download and calibration of sigma and rho
tests/                  pytest suite
results/                generated tables and figures (committed)
docs/                   methods, results, changelog
.github/workflows/      GitHub Actions: fast tests on push
```

## References

- Andersen, L., Broadie, M. (2004). Primal-dual simulation algorithm for pricing multidimensional
  American options. *Management Science* 50(9), 1222–1234.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer, Ch. 8.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal
  of Statistics* 6(2), 65–70.
- Johnson, H. (1987). Options on the maximum or the minimum of several assets. *Journal of
  Financial and Quantitative Analysis* 22(3), 277–283.
- Longstaff, F. A., Schwartz, E. S. (2001). Valuing American options by simulation: a simple
  least-squares approach. *Review of Financial Studies* 14(1), 113–147.
- Margrabe, W. (1978). The value of an option to exchange one asset for another. *Journal of
  Finance* 33(1), 177–186.
- Morgan, W. A. (1939). A test for the significance of the difference between the two variances
  in a sample from a normal bivariate population. *Biometrika* 31(1/2), 13–19.
- Pitman, E. J. G. (1939). A note on normal correlation. *Biometrika* 31(1/2), 9–12.
- Stulz, R. (1982). Options on the minimum or the maximum of two risky assets. *Journal of
  Financial Economics* 10(2), 161–185.
- Tsitsiklis, J. N., Van Roy, B. (2001). Regression methods for pricing complex American-style
  options. *IEEE Transactions on Neural Networks* 12(4), 694–703.
