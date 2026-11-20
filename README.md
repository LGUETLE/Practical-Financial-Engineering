# Bermudan Max-Call Pricing with Regression-Based Monte Carlo

Monte Carlo pricing of a Bermudan call on the maximum of two assets with regression-based methods.
The project replicates Example 8.6.1 of Glasserman (2003) and compares four ways of estimating the
continuation value. Classic Longstaff–Schwartz (LSM) and Tsitsiklis–Van Roy (TVR) are two corners
of a 2x2 design: regression target (realized cash flow vs value iteration) x regression sample
(in-the-money paths vs all paths). Every learned exercise policy is priced out of sample, which
gives a low-biased estimate. A duality-based upper bound (Andersen–Broadie) turns each estimate into
a price interval. The European closed form (Stulz 1982) serves as a control variate and as an exact
reference in calibrated scenarios.

Main findings (details and all numbers below):

- **Price intervals.** The default policy gives 95% intervals of [7.94, 8.12], [13.78, 14.06]
  and [21.14, 21.49] for S0 = 90 / 100 / 110. All three contain the Glasserman reference values
  8.08 / 13.90 / 21.34. A policy trained on 100,000 paths narrows the duality gap to 0.04–0.12.
- **LSM vs TVR.** On the benchmark the difference is small (about 0.02 on 13.8). It comes from the
  **regression sample** (all paths vs ITM paths), not from the regression target. With δ = 0 and
  high volatility the **target** matters instead: at σ = 1.0–1.2, value iteration cuts the bias
  against the exact price by more than half.
- **No variance advantage for either method.** The variances of LSM and TVR are statistically
  indistinguishable, also at low path counts and σ = 0.8. The difference at high volatility is in
  the bias (policy quality), not the variance.
- **Basis.** Scaling the basis by the strike lowers the condition number by 3.5–5.4 orders of
  magnitude but changes no price. A complete degree-3 basis is *worse* than degree 2 at 4,000
  training paths.

## Background and changes

Originally developed as a group project (three students) in the Project in Business Mathematics,
Institute of Financial Mathematics, University of Ulm, July 2026. I led the project and wrote the
code. Since then I have revised and extended it individually. Changes compared with the course
version:

- **Statistics.** Every price comes with a standard error or confidence interval. Methods are
  compared on common random numbers with paired tests, and a Holm correction is applied when
  several comparisons are tested together.
- **Reproducibility.** All randomness comes from `np.random.default_rng(seed)` generators passed
  explicitly to the simulation functions (no global NumPy seed). The first stage of this revision
  reported a significant TVR − LSM difference (p = 0.0008) from a single 50-run experiment. On the
  new random stream that experiment gave −0.004 (p = 0.69). A separate confirmation run with 200
  replications and an independent seed gives +0.023, SE 0.005 (see the 2x2 results).
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

## Model and methods

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

## Results

All numbers come from one run of `python run_all.py` (Python 3.13.2, numpy 2.5.5, scipy 1.18.1;
total runtime 97 s on an Apple M4, see [`results/run_info.md`](results/run_info.md)). Full tables
are in [`results/`](results/).

### Glasserman benchmark: price bounds

The lower bound L uses 200,000 independent paths. The upper bound U = L + gap uses 4,000 outer
paths and 2,000 inner paths per in-the-money date. All policies at one S0 share the same random
numbers. Full table, including classic LSM and a policy trained on 500 paths:
[`results/duality_bounds.md`](results/duality_bounds.md).

| S0 | Policy | L (SE) | U (SE) | Gap (SE) | 95% interval | Reference |
|---:|---|---:|---:|---:|---|---:|
| 90 | TVR, 4,000 training paths (default) | 7.9972 (0.0278) | 8.0679 (0.0282) | 0.0707 (0.0047) | [7.9427, 8.1232] | 8.08 |
| 90 | TVR, 100,000 training paths | 8.0111 (0.0276) | 8.0552 (0.0278) | 0.0441 (0.0034) | [7.9571, 8.1097] | 8.08 |
| 100 | TVR, 4,000 training paths (default) | 13.8472 (0.0342) | 13.9869 (0.0354) | 0.1397 (0.0089) | [13.7801, 14.0563] | 13.90 |
| 100 | TVR, 100,000 training paths | 13.8743 (0.0350) | 13.9549 (0.0355) | 0.0806 (0.0058) | [13.8056, 14.0245] | 13.90 |
| 110 | TVR, 4,000 training paths (default) | 21.2174 (0.0405) | 21.4043 (0.0415) | 0.1868 (0.0090) | [21.1381, 21.4855] | 21.34 |
| 110 | TVR, 100,000 training paths | 21.2530 (0.0406) | 21.3704 (0.0411) | 0.1174 (0.0067) | [21.1735, 21.4510] | 21.34 |

- **The reference lies inside the 95% interval in all 12 cases** (four policies x three S0).
- **A better policy gives a smaller gap.** Ranked by lower bound, the four policies have strictly
  increasing gaps at every S0. At S0 = 100: 0.081 (TVR, 100,000 paths) < 0.140 (TVR, 4,000)
  < 0.161 (classic LSM, 4,000) < 0.442 (TVR, 500).
- **The gap separates policies more sharply than L.** At S0 = 110 classic LSM and TVR have
  indistinguishable lower bounds (difference −0.0095, SE 0.0118). The LSM gap is nevertheless
  larger by 0.025 (SE 0.008, Holm-adjusted p = 0.002).

### Four regression methods (2x2 design)

R = 50 replications per S0, N = 4,000 training and 4,000 pricing paths; all methods use the same
paths. Shown: mean (SE over replications). Full tables:
[`results/benchmark_2x2.md`](results/benchmark_2x2.md).

| S0 | Reference | Classic LSM | Cash flow, all | Value, ITM | TVR |
|---:|---:|---:|---:|---:|---:|
| 90 | 8.08 | 8.0059 (0.0285) | 8.0242 (0.0274) | 8.0148 (0.0272) | 8.0201 (0.0289) |
| 100 | 13.90 | 13.8442 (0.0322) | 13.8729 (0.0321) | 13.8619 (0.0324) | 13.8684 (0.0331) |
| 110 | 21.34 | 21.2448 (0.0381) | 21.2542 (0.0381) | 21.2449 (0.0349) | 21.2464 (0.0374) |

All twelve means lie below the reference, as expected for low-biased estimators; biases range from
−0.03 to −0.10. (Cash flow, all paths) has the highest mean at every S0. After a Holm correction
over the nine comparisons with it, only classic LSM at S0 = 100 is significantly worse (−0.029,
adjusted p = 0.006).

To identify the driving factor, a confirmation run was made at S0 = 100: R = 200, an independent
seed, and four paired tests fixed before the run, with Holm correction.

| Contrast | Estimate | SE | Holm p |
|---|---:|---:|---:|
| TVR − classic LSM | +0.0226 | 0.0052 | 6.7e-05 |
| Target (value − cash flow) | −0.0003 | 0.0041 | 0.94 |
| Sample (all − ITM) | +0.0229 | 0.0034 | 1.1e-09 |
| Interaction | +0.0049 | 0.0052 | 0.69 |

On this benchmark **the regression sample, not the target, drives the LSM/TVR difference.**
- Fitting on all paths instead of ITM paths only gives a better policy (+0.023).
- Value-iteration targets instead of realized cash flows make no detectable difference.
- TVR − LSM is the sum of the two main effects, so it is almost entirely the sample effect.
- The effect is small: about 0.15% of the price.

### Classic LSM vs TVR: variances

On the 50 replications at S0 = 100 ([`results/lsm_vs_tvr.md`](results/lsm_vs_tvr.md)), the
variances are 0.0517 (LSM) and 0.0549 (TVR). Neither the F-test (p = 0.83) nor the Pitman–Morgan
test for paired samples (p = 0.47) rejects equal variances. The two price series are strongly
correlated across replications (0.96), as both use the same paths; the F-test ignores this, the
Pitman–Morgan test accounts for it. The paired mean difference TVR − LSM is +0.0243 (SE 0.0094,
p = 0.013), consistent with the R = 200 run above.

### Basis ablation

S0 = 100, R = 50, all eight configurations on the same paths
([`results/basis_ablation.md`](results/basis_ablation.md)).

| Method | Basis | Mean (SE) | Bias | median log10 condition number, raw → S/K |
|---|---|---:|---:|---|
| Classic LSM | degree 2 | 13.8117 (0.0337) | −0.088 | 6.0 → 2.3 |
| Classic LSM | degree 3 | 13.7831 (0.0319) | −0.117 | 8.7 → 3.3 |
| TVR | degree 2 | 13.8340 (0.0311) | −0.066 | 5.6 → 2.1 |
| TVR | degree 3 | 13.8218 (0.0305) | −0.078 | 8.3 → 3.0 |

- **Scaling** lowers the median condition number by 3.5–5.4 orders of magnitude but gives
  bit-identical prices on all 50 runs. In double precision the raw basis is not ill-conditioned
  enough to change a single exercise decision.
- **Degree 3** gives a lower mean than degree 2: paired difference −0.0286 for LSM (Holm p = 0.011)
  and −0.0123 for TVR (p = 0.14). With 4,000 training paths the four extra coefficients add more
  estimation noise to the policy than they remove approximation error.
- **Default.** Both estimators are low-biased, so degree 2 (scaled) is the default.

### European closed form and control variate

**Closed form.** The Stulz price agrees with a direct Monte Carlo price (N = 2,000,000) within 1.45
standard errors on all seven parameter sets
([`results/stulz_check.md`](results/stulz_check.md)). The sets cover ρ ∈ {0, ±0.5, 0.3, 0.8, 0.9,
−0.9} and unequal volatilities, spot prices and dividend yields.

**Control variate.** Applied to the TVR policy on the same 50 replications as above
([`results/control_variate.md`](results/control_variate.md)):

| S0 | European (Stulz) | Mean plain (SE) | Mean with CV (SE) | Corr(Y, X) | VRF per run | Variance ratio across runs |
|---:|---:|---:|---:|---:|---:|---:|
| 90 | 6.6551 | 8.0201 (0.0289) | 8.0186 (0.0217) | 0.69 | 1.92 | 1.78 |
| 100 | 11.1957 | 13.8684 (0.0331) | 13.8650 (0.0266) | 0.65 | 1.71 | 1.56 |
| 110 | 16.9286 | 21.2464 (0.0374) | 21.2390 (0.0307) | 0.60 | 1.56 | 1.49 |

- **The variance reduction is modest** (factor 1.5–1.9). With δ = 0.10 early exercise is worth a
  lot (13.90 vs 11.20 at S0 = 100), so the Bermudan and European payoffs are only moderately
  correlated.
- **Most of the variance across replications is pricing noise.** The reduction across runs, which
  also contains the training noise of the policy, is almost as large as the per-run factor.

### Calibrated scenarios: bias against an exact reference

**Setup.**
- Data: daily adjusted closes from Yahoo Finance, 2022-01-01 to 2025-12-31, for both pairs;
  annualised with 252 periods per year for stocks and 365 for crypto.
- Contract: S0 = 100 for both assets (normalised), K = 100, T = 1, r = 0.05, 12 exercise dates,
  δ = 0.
- Since δ = 0, the Stulz price is the exact Bermudan price.
- Each method: R = 50 replications with N = 10,000 paths, all methods on the same paths.
- Full tables: [`results/empirical.md`](results/empirical.md).

Bias = mean − Stulz, with the SE over replications in brackets:

| Scenario | σ_1 / σ_2 / ρ | Stulz | Classic LSM | Cash flow, all | Value, ITM | TVR |
|---|---|---:|---:|---:|---:|---:|
| AAPL / MSFT | 0.284 / 0.267 / 0.626 | 19.2203 | −0.052 (0.040) | −0.036 (0.039) | −0.039 (0.040) | −0.032 (0.039) |
| BTC / ETH | 0.515 / 0.698 / 0.838 | 35.3960 | −0.310 (0.106) | −0.235 (0.106) | −0.192 (0.106) | −0.162 (0.106) |

- **All means lie below the exact price.** The RMSE over replications is 0.28 (tech) and 0.76–0.80
  (crypto).
- **Only classic LSM in the crypto scenario is significantly biased** after a Holm correction over
  the eight tests (p = 0.04).
- **The policies stop 6–10% of the pricing paths early**, although stopping early is never optimal
  here. Most of these paths are deep in the money, where the loss from stopping early is small.

**Volatility sweep.** A sweep over a common volatility (σ_1 = σ_2 = σ, ρ = 0.5, R = 20 per σ)
shows the bias growing with σ and the methods separating at high σ:
- At σ = 1.2 the bias is −2.14 for classic LSM and −0.70 for TVR (paired difference +1.44,
  SE 0.31, Holm p = 0.003).
- Unlike on the Glasserman benchmark, here the **regression target** matters. Value iteration
  instead of realized cash flows improves the price by 0.09 at σ = 0.6 (Holm p = 0.005) and by 1.15
  at σ = 1.2 (Holm p = 0.018).
- The sample effect is not significant at any σ after the correction.
- A plausible explanation: with very noisy payoffs, the realized cash-flow targets give noisy fits
  in the money and hence more spurious early exercise. Note that the significant target effect
  appears only at σ ≥ 0.6. At the calibrated volatilities the methods differ by less than their
  standard errors.

![Bias against the Stulz price vs volatility](results/empirical_sigma_sensitivity.png)

### Stress test: low path counts at σ = 0.8

Glasserman parameters with σ = 0.8, N ∈ {250, …, 4,000} training and pricing paths, R = 50
replications per N, all methods on the same paths. Classic LSM is also run with the raw (unscaled)
basis. Full table: [`results/stresstest.md`](results/stresstest.md).

| N | Classic LSM mean (variance) | TVR mean (variance) | Var LSM / Var TVR | Holm p (variance) | TVR − LSM (SE) |
|---:|---:|---:|---:|---:|---:|
| 250 | 65.59 (100.2) | 71.33 (118.2) | 0.85 | 1.00 | +5.74 (1.16) |
| 500 | 67.06 (42.9) | 70.74 (45.0) | 0.95 | 1.00 | +3.68 (0.77) |
| 1,000 | 68.36 (37.2) | 71.42 (35.7) | 1.04 | 1.00 | +3.06 (0.46) |
| 2,000 | 69.26 (15.6) | 71.57 (14.3) | 1.09 | 1.00 | +2.32 (0.40) |
| 4,000 | 69.95 (9.7) | 71.61 (7.4) | 1.32 | 0.91 | +1.67 (0.32) |

- **No LSM variance blow-up.** The original stress test was written to show one, using single
  unseeded runs with R = 30. With seeds, R = 50 and paired tests, the variances of LSM and TVR are
  statistically indistinguishable at every N (Pitman–Morgan test, Holm-corrected). Both fall
  roughly like 1/N.
- **Scaling does not matter here either.** Raw and scaled LSM give identical prices at every N,
  although the raw condition numbers reach 10^8.
- **What differs is the bias.** LSM's mean is lower by 1.7–5.7 (all Holm p < 1e-4), so its policy
  is clearly worse at high volatility and small N.
- **The basis is weak at this volatility.** The duality interval of a TVR policy trained on 100,000
  paths is [71.97, 75.51] (L = 72.61, U = 74.84). Its gap of 2.2 (3% of the price) shows that the
  degree-2 basis approximates the continuation value much less well at σ = 0.8 than at σ = 0.2.

![Variance vs N](results/stresstest_variance.png)

### Estimated exercise region

The estimated exercise region at exercise date i = 4 (t = 4/3) comes from a TVR policy trained on
20,000 paths. The policy exercises when one asset is well above the strike and clearly above the
other, and continues near the diagonal S_1 = S_2, as in Glasserman's Figure 8.9.

![Estimated exercise region at step 4](results/exercise_boundary.png)

Single runs (one training and one pricing set) for S0, σ and N are in
[`results/glasserman_replication.md`](results/glasserman_replication.md) and
[`results/single_runs.md`](results/single_runs.md). Their deviations from the reference, up to
1.3 standard errors, are Monte Carlo noise.

### Limitations

- The reference values 8.08 / 13.90 / 21.34 are quoted to two decimals; biases of a few hundredths
  are therefore only resolved through the duality interval, not against the reference alone.
- The bounds and their paired comparisons are conditional on one trained policy per S0. The 2x2
  replications include the training noise; the bounds do not.
- The calibrated scenarios use a constant-volatility GBM. They test the estimators at realistic
  volatility and correlation levels; they are not a market price of any traded product.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py              # regenerates results/ (about 100 s on an Apple M4)
```

`run_all.py` runs the four experiment scripts in order and writes the runtimes to
`results/run_info.md`. The scripts can also be run on their own (from the repository root):

| Script | Content | Runtime |
|---|---|---:|
| `theory_engine.py` | basis ablation, 2x2 design, LSM vs TVR, Stulz check, control variate, single runs, exercise region | 7 s |
| `bounds_engine.py` | lower and upper bounds by duality | 72 s |
| `empirical_engine.py` | calibration, bias against the Stulz price, volatility sweep | 9 s |
| `stresstest_engine.py` | low path counts at σ = 0.8 | 8 s |

`empirical_engine.py` downloads prices from Yahoo Finance on the first run (network access needed)
and caches them in `data/`. LaTeX/CSV tables go to `output_tables/` and additional plots to
`output_plots/`; both are not committed.

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
maxcall/
  simulation.py         exact GBM simulation (independent and correlated)
  lsm.py                basis, the four regression methods, forward pass, control variate
  european.py           Stulz closed form, Margrabe, Black-Scholes, direct Monte Carlo
  duality.py            Andersen-Broadie upper bound with nested simulation
  mc_stats.py           summaries, paired tests, Pitman-Morgan, Holm, Markdown tables
  data.py               Yahoo Finance download and calibration of sigma and rho
tests/                  pytest suite
results/                tables and figures shown in this README (generated, committed)
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
