"""Shared fixtures: Glasserman benchmark parameters and small trained models."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BENCH = dict(K=100.0, T=3.0, r=0.05, delta=0.10, sigma=0.20, m=9)


@pytest.fixture
def bench() -> dict:
    return dict(BENCH)


@pytest.fixture
def small_paths(bench):
    """Training and pricing paths at S0 = 100 (1,000 each) from a fixed seed."""
    from maxcall.simulation import simulate_gbm
    rng = np.random.default_rng(12345)
    b = bench
    args = (100.0, b["sigma"], b["r"], b["delta"], b["T"], b["m"])
    return tuple(simulate_gbm(*args, 1000, rng) for _ in range(4))
