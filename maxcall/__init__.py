"""
maxcall: regression-based Monte Carlo pricing of a Bermudan call on the maximum of two assets.

Modules
  simulation  exact GBM simulation (independent and correlated assets)
  lsm         regression basis, 2x2 regression methods (LSM, TVR, ...), forward pass, control variate
  european    Stulz (1982) closed form for the European max-call, Margrabe limit, Black-Scholes
  duality     Andersen-Broadie primal-dual upper bound with nested simulation
  mc_stats    summaries, paired tests, Holm correction, Markdown tables
  data        Yahoo Finance prices and calibration of volatilities and correlation
"""
from .european import stulz_max_call
from .lsm import (DEFAULT_DEGREE, DEFAULT_SCALE, METHODS, apply_policy, basis_max,
                  lsm_price_max, lsm_price_max_cv, lsm_train_max)
from .simulation import simulate_correlated_gbm, simulate_gbm

__all__ = ["DEFAULT_DEGREE", "DEFAULT_SCALE", "METHODS", "apply_policy", "basis_max",
           "lsm_price_max", "lsm_price_max_cv", "lsm_train_max", "simulate_correlated_gbm",
           "simulate_gbm", "stulz_max_call"]
