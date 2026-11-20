"""
Statistical helpers for Monte Carlo experiments: summaries across replications, paired tests
under common random numbers, Holm correction, and Markdown table output.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.stats as stats


def summarize(prices: np.ndarray, reference: float | None = None) -> dict:
    """
    Summary of R replicated price estimates: mean, standard error of the mean (ddof=1),
    variance, 95% t-confidence interval and, if a reference value is given, the bias
    (mean - reference) and the RMSE sqrt(mean((price - reference)^2)).
    """
    x = np.asarray(prices, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    tq = stats.t.ppf(0.975, n - 1)
    out = dict(mean=mean, se=se, var=float(np.var(x, ddof=1)),
               ci_low=mean - tq * se, ci_high=mean + tq * se, n=n)
    if reference is not None:
        out["bias"] = mean - reference
        out["rmse"] = float(np.sqrt(np.mean((x - reference) ** 2)))
    return out


def paired_t(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Paired t-test of H0: E[x - y] = 0 for estimates computed on common random numbers.
    Returns mean difference, its standard error, t statistic, two-sided p-value and 95% CI.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    if se == 0.0:
        return dict(diff=mean, se=0.0, t=np.nan, p=np.nan, ci_low=mean, ci_high=mean, n=n)
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), n - 1))
    tq = stats.t.ppf(0.975, n - 1)
    return dict(diff=mean, se=se, t=float(t), p=p,
                ci_low=mean - tq * se, ci_high=mean + tq * se, n=n)


def pitman_morgan(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Pitman-Morgan test of H0: Var(x) = Var(y) for paired samples, via the correlation of
    x + y and x - y (zero under H0). Returns variance ratio, r and two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = stats.pearsonr(x + y, x - y)
    return dict(var_ratio=float(np.var(x, ddof=1) / np.var(y, ddof=1)), r=float(r), p=float(p))


def holm(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1). NaN stays NaN."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    k = len(pv)
    order = np.argsort(pv)
    running = 0.0
    out = np.empty(k)
    for rank, idx in enumerate(order):
        running = max(running, (k - rank) * pv[idx])
        out[idx] = min(running, 1.0)
    adj[ok] = out
    return adj


def fmt_p(p: float) -> str:
    """Format a p-value: 4 decimals, or scientific notation below 1e-4."""
    if p is None or np.isnan(p):
        return "–"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def df_to_markdown(df: pd.DataFrame, floatfmt: dict | str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub Markdown table (numbers right-aligned)."""
    cols = list(df.columns)
    def fmt(col: str, v: object) -> str:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "–"
            f = floatfmt.get(col, "{:.4f}") if isinstance(floatfmt, dict) else floatfmt
            return f(v) if callable(f) else f.format(v)
        return str(v)
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in cols]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(c, row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_result(name: str, text: str, folder: str = "results") -> str:
    """Write a Markdown result file to results/<name>.md and return its path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    return path
