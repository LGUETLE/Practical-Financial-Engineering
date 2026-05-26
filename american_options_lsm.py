"""
==============================================================================
American Options Pricing via Regression-Based Methods (Longstaff-Schwartz)
==============================================================================
Seminar: American Options – Regression-Based Methods
Basis:   Glasserman (2003), Chapter 8.6

Struktur:
  1. Datenbeschaffung  – echte Kurse via yfinance ODER synthetische GBM-Daten
  2. Parameterkalibrierung  – σ, μ aus historischen Log-Returns
  3. GBM-Pfadsimulation
  4. LSM-Algorithmus  – American Put (1 Asset) & American Max-Call (2 Assets)
  5. Binomial-Lattice als Benchmark
  6. Visualisierungen
     – simulierte Pfade
     – geschätzte Continuation Values
     – Ausübungsregion (Max-Option, 2D)
     – Konvergenz in Pfadanzahl

Abhängigkeiten:
  pip install numpy scipy matplotlib pandas yfinance
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.linalg import lstsq

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION  –  hier alles anpassen
# ─────────────────────────────────────────────────────────────────────────────

USE_REAL_DATA   = True    # True → yfinance;  False → synthetische GBM-Daten

# Ticker für echte Daten (USE_REAL_DATA=True)
TICKER_1        = "AAPL"   # Asset 1  (Put & Max-Option)
TICKER_2        = "MSFT"   # Asset 2  (nur Max-Option)
DATA_PERIOD     = "3y"     # yfinance-Periode

# Fallback-Parameter wenn USE_REAL_DATA=False  (angelehnt an Glasserman 8.6.1)
SIGMA_DEFAULT   = 0.20     # Volatilität
MU_DEFAULT      = 0.05     # Drift (= r bei risikoneutraler Welt)

# Optionsparameter (Glasserman Beispiel 8.6.1 & 8.6.3)
S0              = 100.0    # Anfangskurs
K               = 100.0    # Strike
T               = 3.0      # Laufzeit in Jahren
r               = 0.05     # risikofreier Zinssatz
DELTA           = 0.10     # Dividendenrendite
M_STEPS         = 9        # Ausübungszeitpunkte (wie Glasserman 8.6.1)
N_PATHS         = 4000     # Anzahl Simulationspfade
N_PATHS_LOW     = 4000     # Pfade für Low Estimator
SEED            = 42

# Basisfunktionen: verfügbare Sets (Index 0..5, entspricht Tabelle 8.1)
# Jedes Element: Liste von lambda-Funktionen  f(s1) oder f(s1,s2)
BASIS_LABELS = [
    "1, S, S², S³",
    "1, S, S², S³, S⁴",
    "1, S, S², h(S)",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATENBESCHAFFUNG & PARAMETERKALIBRIERUNG
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_calibrate(ticker: str, period: str = "3y") -> dict:
    """
    Lädt historische Kursdaten via yfinance und schätzt
    annualisierte Volatilität σ und Drift μ.

    Returns dict mit keys: sigma, mu, S_last, ticker, prices
    """
    try:
        import yfinance as yf
        data  = yf.Ticker(ticker).history(period=period)
        if data.empty:
            raise ValueError(f"Keine Daten für {ticker}")
        prices    = data["Close"].dropna()
        log_ret   = np.log(prices / prices.shift(1)).dropna()
        sigma     = float(log_ret.std()  * np.sqrt(252))
        mu        = float(log_ret.mean() * 252)
        S_last    = float(prices.iloc[-1])
        print(f"  [{ticker}]  σ={sigma:.4f}  μ={mu:.4f}  "
              f"S_last={S_last:.2f}  n={len(prices)} Handelstage")
        return dict(sigma=sigma, mu=mu, S_last=S_last,
                    ticker=ticker, prices=prices)
    except Exception as exc:
        print(f"  Warnung: yfinance für {ticker} fehlgeschlagen ({exc}). "
              f"Nutze Standardparameter.")
        return dict(sigma=SIGMA_DEFAULT, mu=MU_DEFAULT,
                    S_last=S0, ticker=ticker, prices=None)


def get_parameters():
    """Gibt (params1, params2) zurück – kalibriert oder synthetisch."""
    if USE_REAL_DATA:
        print("── Lade Marktdaten ──────────────────────────────────")
        p1 = fetch_and_calibrate(TICKER_1, DATA_PERIOD)
        p2 = fetch_and_calibrate(TICKER_2, DATA_PERIOD)
        print()
    else:
        print("── Synthetische GBM-Parameter (Glasserman 8.6) ──────")
        p1 = dict(sigma=SIGMA_DEFAULT, mu=r, S_last=S0, ticker="Asset1")
        p2 = dict(sigma=SIGMA_DEFAULT, mu=r, S_last=S0, ticker="Asset2")
        print(f"  σ={SIGMA_DEFAULT}  r={r}  δ={DELTA}")
        print()
    return p1, p2


# ─────────────────────────────────────────────────────────────────────────────
# 2. GBM-PFADSIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_gbm(S0: float, sigma: float, r: float, delta: float,
                 T: float, m: int, n_paths: int,
                 seed: int = None) -> np.ndarray:
    """
    Simuliert n_paths GBM-Pfade mit m Zeitschritten.

    Risikoneutrale Dynamik:
        S(t+dt) = S(t) * exp((r - δ - σ²/2)*dt + σ*√dt*Z)

    Returns:
        paths: array (n_paths, m+1)  –  paths[:,0] = S0
    """
    if seed is not None:
        np.random.seed(seed)
    dt     = T / m
    drift  = (r - delta - 0.5 * sigma**2) * dt
    vol    = sigma * np.sqrt(dt)
    Z      = np.random.standard_normal((n_paths, m))
    log_S  = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    S      = np.hstack([np.full((n_paths, 1), S0),
                        np.exp(log_S)])
    return S


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASISFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def basis_put(S: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Basisfunktionen für den American Put (1 Asset).
    Laguerre-Polynome (wie Longstaff-Schwartz [241]) oder einfache Potenzen.

    Args:
        S:      (n,) Vektor von Asset-Preisen
        degree: maximaler Polynomgrad

    Returns:
        Φ: (n, degree+1) Design-Matrix
    """
    # Normierung stabilisiert die Regression
    s = S / S.mean()
    cols = [np.ones(len(s))]
    for d in range(1, degree + 1):
        cols.append(s**d)
    return np.column_stack(cols)


def basis_max(S1: np.ndarray, S2: np.ndarray,
              include_h: bool = True) -> np.ndarray:
    """
    Basisfunktionen für die Max-Option (2 Assets), analog Tabelle 8.1.

    Enthält: 1, S1, S2, S1², S2², S1·S2, [h(S1,S2)]
    """
    n   = len(S1)
    s1  = S1 / S1.mean()
    s2  = S2 / S2.mean()
    cols = [
        np.ones(n),
        s1,
        s2,
        s1**2,
        s2**2,
        s1 * s2,
    ]
    if include_h:
        h = np.maximum(np.maximum(S1, S2) - K, 0)
        cols.append(h / (h.mean() + 1e-8))
    return np.column_stack(cols)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSM-ALGORITHMUS
# ─────────────────────────────────────────────────────────────────────────────

def lsm_put(paths: np.ndarray, K: float, r: float, T: float,
            degree: int = 3) -> tuple[float, np.ndarray]:
    """
    Longstaff-Schwartz Algorithmus für den American Put (1 Asset).

    Implementierung nach Glasserman (8.51)/(8.52):
      – Backward Induction über alle Ausübungszeitpunkte
      – Regression der Continuation Values auf Basisfunktionen
      – Ausübung wenn h(S) ≥ Ĉ(S)

    Args:
        paths:  (n_paths, m+1) GBM-Pfade
        K, r, T: Optionsparameter
        degree: Polynomgrad der Basisfunktionen

    Returns:
        price:          LSM-Preisschätzung
        exercise_idx:   (n_paths,) Ausübungszeitpunkt je Pfad
    """
    n_paths, m1 = paths.shape
    m   = m1 - 1
    dt  = T / m
    df  = np.exp(-r * dt)      # Diskontierungsfaktor pro Schritt

    # Cashflows initialisieren: terminale Auszahlung
    cashflow   = np.maximum(K - paths[:, -1], 0.0)
    exercise_t = np.full(n_paths, m, dtype=int)   # Zeitpunkt der Ausübung

    # ── Backward Induction ───────────────────────────────────────────────────
    for i in range(m - 1, 0, -1):
        S_i        = paths[:, i]
        h_i        = np.maximum(K - S_i, 0.0)     # sofortiger Payoff

        # Nur in-the-money Pfade für Regression (Empfehlung Longstaff-Schwartz)
        itm        = h_i > 0
        if itm.sum() < 5:
            cashflow *= df
            continue

        # Diskontierte zukünftige Cashflows
        pv_future  = cashflow[itm] * df**(exercise_t[itm] - i)

        # Regression: E[PV | S_i]
        Phi        = basis_put(S_i[itm], degree)
        beta, _, _, _ = lstsq(Phi, pv_future)
        C_hat      = Phi @ beta                   # geschätzte Continuation Value

        # Ausübungsentscheidung
        exercise   = h_i[itm] >= np.maximum(C_hat, 0)
        idx        = np.where(itm)[0]
        cashflow[idx[exercise]]   = h_i[itm][exercise]
        exercise_t[idx[exercise]] = i
        cashflow[idx[~exercise]] *= df

    # Diskontierung auf t=0
    price = np.mean(cashflow * df**exercise_t)
    return price, exercise_t


def lsm_max_option(paths1: np.ndarray, paths2: np.ndarray,
                   K: float, r: float, T: float,
                   include_h: bool = True) -> tuple[float, np.ndarray, list]:
    """
    LSM für American Max-Call Option auf 2 Assets (Glasserman Beispiel 8.6.1).

    Payoff: h̃(S1, S2) = (max(S1, S2) - K)⁺

    Returns:
        price:          LSM-Preisschätzung
        exercise_idx:   (n_paths,) Ausübungszeitpunkt
        betas:          Liste der β-Vektoren (für Interpretation)
    """
    n_paths, m1 = paths1.shape
    m   = m1 - 1
    dt  = T / m
    df  = np.exp(-r * dt)

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K, 0.0)

    cashflow   = payoff(paths1[:, -1], paths2[:, -1])
    exercise_t = np.full(n_paths, m, dtype=int)
    betas      = []

    for i in range(m - 1, 0, -1):
        S1_i = paths1[:, i]
        S2_i = paths2[:, i]
        h_i  = payoff(S1_i, S2_i)

        itm  = h_i > 0
        if itm.sum() < 10:
            cashflow *= df
            continue

        pv_future = cashflow[itm] * df**(exercise_t[itm] - i)

        Phi       = basis_max(S1_i[itm], S2_i[itm], include_h)
        beta, _, _, _ = lstsq(Phi, pv_future)
        betas.append(beta)

        C_hat     = np.maximum(Phi @ beta, 0)
        exercise  = h_i[itm] >= C_hat
        idx       = np.where(itm)[0]
        cashflow[idx[exercise]]   = h_i[itm][exercise]
        exercise_t[idx[exercise]] = i
        cashflow[idx[~exercise]] *= df

    price = np.mean(cashflow * df**exercise_t)
    return price, exercise_t, betas


# ─────────────────────────────────────────────────────────────────────────────
# 5. BINOMIAL-LATTICE (Benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def binomial_american_put(S0: float, K: float, r: float, sigma: float,
                          delta: float, T: float, N: int = 500) -> float:
    """
    Binomial-Lattice für American Put (CRR-Parametrisierung).
    Dient als Benchmark für den LSM-Algorithmus.

    N = 500 Schritte liefert auf 4 Stellen genaue Werte.
    """
    dt  = T / N
    u   = np.exp(sigma * np.sqrt(dt))
    d   = 1.0 / u
    p   = (np.exp((r - delta) * dt) - d) / (u - d)
    df  = np.exp(-r * dt)

    # Terminale Aktienpreise
    j   = np.arange(N + 1)
    S_T = S0 * u**j * d**(N - j)
    V   = np.maximum(K - S_T, 0.0)

    # Backward Induction
    for _ in range(N):
        V   = df * (p * V[1:] + (1 - p) * V[:-1])
        j   = np.arange(len(V))
        S_t = S0 * u**j * d**(len(V) - 1 - j)
        V   = np.maximum(V, np.maximum(K - S_t, 0.0))

    return float(V[0])


# ─────────────────────────────────────────────────────────────────────────────
# 6. KONVERGENZANALYSE
# ─────────────────────────────────────────────────────────────────────────────

def convergence_analysis(sigma: float, path_counts=None) -> dict:
    """
    Berechnet LSM-Preise für verschiedene Pfadanzahlen N_paths
    und vergleicht mit Binomial-Benchmark.
    """
    if path_counts is None:
        path_counts = [200, 500, 1000, 2000, 4000, 8000]

    true_price = binomial_american_put(S0, K, r, sigma, DELTA, T, N=1000)
    results    = {"n_paths": [], "price": [], "std": [], "true": true_price}

    for n in path_counts:
        reps   = 20
        prices = []
        for rep in range(reps):
            paths = simulate_gbm(S0, sigma, r, DELTA, T, M_STEPS, n,
                                 seed=rep * 1000)
            p, _  = lsm_put(paths, K, r, T, degree=3)
            prices.append(p)
        results["n_paths"].append(n)
        results["price"].append(np.mean(prices))
        results["std"].append(np.std(prices))
        print(f"  N={n:5d}  Ø={np.mean(prices):.4f}  "
              f"σ={np.std(prices):.4f}  "
              f"Benchmark={true_price:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISIERUNGEN
# ─────────────────────────────────────────────────────────────────────────────

def plot_gbm_paths(paths: np.ndarray, title: str = "GBM-Pfade",
                  n_show: int = 50, ax=None):
    """Zeigt simulierte GBM-Pfade."""
    show_ax = ax is None
    if show_ax:
        fig, ax = plt.subplots(figsize=(9, 4))
    t = np.linspace(0, T, paths.shape[1])
    for i in range(min(n_show, paths.shape[0])):
        ax.plot(t, paths[i], lw=0.6, alpha=0.4, color="steelblue")
    ax.axhline(K, color="crimson", lw=1.5, ls="--", label=f"Strike K={K}")
    ax.set_xlabel("Zeit (Jahre)")
    ax.set_ylabel("Aktienkurs")
    ax.set_title(title)
    ax.legend()
    if show_ax:
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/01_gbm_paths.png", dpi=150)
        plt.show()


def plot_continuation_value(paths: np.ndarray, K: float, r: float,
                            T: float, i_step: int = 4, degree: int = 3):
    """
    Visualisiert die geschätzte Continuation Value Ĉ_i(S)
    gegenüber der sofortigen Ausübung h(S) = (K-S)⁺.
    """
    # Kurze LSM-Runde um β_i zu erhalten
    n_paths, m1 = paths.shape
    m   = m1 - 1
    dt  = T / m

    cashflow   = np.maximum(K - paths[:, -1], 0.0)
    exercise_t = np.full(n_paths, m, dtype=int)
    beta_at_i  = None

    for i in range(m - 1, 0, -1):
        S_i  = paths[:, i]
        h_i  = np.maximum(K - S_i, 0.0)
        itm  = h_i > 0
        df   = np.exp(-r * (T / m))

        if itm.sum() >= 5:
            pv   = cashflow[itm] * df**(exercise_t[itm] - i)
            Phi  = basis_put(S_i[itm], degree)
            beta, _, _, _ = lstsq(Phi, pv)
            if i == i_step:
                beta_at_i = beta
            C_hat = np.maximum(Phi @ beta, 0)
            exc   = h_i[itm] >= C_hat
            idx   = np.where(itm)[0]
            cashflow[idx[exc]]    = h_i[itm][exc]
            exercise_t[idx[exc]]  = i
            cashflow[idx[~exc]]  *= df
        else:
            cashflow *= df

    if beta_at_i is None:
        print("Warnung: β nicht gefunden für Schritt", i_step)
        return

    S_range = np.linspace(60, 140, 300)
    Phi_r   = basis_put(S_range, degree)
    C_range = Phi_r @ beta_at_i
    h_range = np.maximum(K - S_range, 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S_range, h_range,  color="crimson",   lw=2, label="h(S) = (K−S)⁺  [sofortige Ausübung]")
    ax.plot(S_range, C_range,  color="steelblue", lw=2, label=f"Ĉ(S)  [Continuation Value, Schritt {i_step}]")
    ax.axvline(K, color="gray", ls=":", lw=1)
    ax.fill_between(S_range, h_range, C_range,
                    where=(h_range >= np.maximum(C_range, 0)),
                    alpha=0.25, color="crimson", label="Ausübungsregion")
    ax.set_xlim(60, 140)
    ax.set_ylim(-2, 42)
    ax.set_xlabel("Aktienkurs S")
    ax.set_ylabel("Wert")
    ax.set_title(f"Estimated Continuation Value vs. Intrinsic Value  (i={i_step})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/02_continuation_value.png", dpi=150)
    plt.show()


def plot_exercise_region_2d(paths1: np.ndarray, paths2: np.ndarray,
                            K: float, r: float, T: float, i_step: int = 4):
    """
    Visualisiert die Ausübungsregion der Max-Option im (S1, S2)-Raum
    analog zu Abbildung 8.9 in Glasserman.
    """
    n_paths, m1 = paths1.shape
    m   = m1 - 1
    dt  = T / m
    df  = np.exp(-r * dt)

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K, 0.0)

    cashflow   = payoff(paths1[:, -1], paths2[:, -1])
    exercise_t = np.full(n_paths, m, dtype=int)
    beta_at_i  = None

    for i in range(m - 1, 0, -1):
        S1_i = paths1[:, i]
        S2_i = paths2[:, i]
        h_i  = payoff(S1_i, S2_i)
        itm  = h_i > 0
        if itm.sum() < 10:
            cashflow *= df
            continue
        pv   = cashflow[itm] * df**(exercise_t[itm] - i)
        Phi  = basis_max(S1_i[itm], S2_i[itm], include_h=True)
        beta, _, _, _ = lstsq(Phi, pv)
        if i == i_step:
            beta_at_i = beta
        C_hat = np.maximum(Phi @ beta, 0)
        exc   = h_i[itm] >= C_hat
        idx   = np.where(itm)[0]
        cashflow[idx[exc]]    = h_i[itm][exc]
        exercise_t[idx[exc]]  = i
        cashflow[idx[~exc]]  *= df

    if beta_at_i is None:
        print("Warnung: β nicht gefunden für Schritt", i_step)
        return

    # Gitter im (S1, S2)-Raum
    s_vals = np.linspace(60, 140, 120)
    S1g, S2g = np.meshgrid(s_vals, s_vals)
    S1f, S2f = S1g.flatten(), S2g.flatten()
    h_grid   = payoff(S1f, S2f)
    Phi_g    = basis_max(S1f, S2f, include_h=True)
    C_grid   = np.maximum(Phi_g @ beta_at_i, 0)
    exercise_region = (h_grid >= C_grid) & (h_grid > 0)
    exercise_region = exercise_region.reshape(S1g.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(S1g, S2g, exercise_region.astype(float),
                levels=[0.5, 1.5], colors=["#6baed6"], alpha=0.5)
    ax.contour(S1g, S2g, exercise_region.astype(float),
               levels=[0.5], colors=["steelblue"], linewidths=2)

    # Optimale Grenze (vereinfacht: max(S1,S2) = K + const)
    # Echte optimale Grenze aus dem Binomial-Modell nicht trivial in 2D;
    # wir zeigen die Diagonale S1=S2 als Referenzlinie
    ax.plot(s_vals, s_vals, "k--", lw=1.5, label="S₁ = S₂ (Symmetrieachse)")
    ax.axhline(K, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(K, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("S₁")
    ax.set_ylabel("S₂")
    ax.set_title(f"Ausübungsregion – American Max-Option  (Schritt i={i_step})\n"
                 f"[Blau = Ausüben, Weiß = Halten]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/03_exercise_region_2d.png", dpi=150)
    plt.show()


def plot_convergence(conv_results: dict):
    """Zeigt LSM-Konvergenz in Pfadanzahl gegen Benchmark."""
    n_arr = np.array(conv_results["n_paths"])
    p_arr = np.array(conv_results["price"])
    s_arr = np.array(conv_results["std"])
    true  = conv_results["true"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(n_arr, p_arr, yerr=1.96 * s_arr, fmt="o-",
                color="steelblue", capsize=4, label="LSM (±1.96σ)")
    ax.axhline(true, color="crimson", ls="--", lw=1.8,
               label=f"Binomial Benchmark ({true:.4f})")
    ax.set_xlabel("Anzahl Simulationspfade")
    ax.set_ylabel("Optionspreis")
    ax.set_title("LSM-Konvergenz in Pfadanzahl")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/04_convergence.png", dpi=150)
    plt.show()


def plot_basis_comparison(paths: np.ndarray, K: float, r: float,
                          T: float, sigma: float):
    """
    Vergleich verschiedener Basisfunktionssets (analog Tabelle 8.1).
    """
    degrees    = [1, 2, 3, 4, 5]
    benchmark  = binomial_american_put(S0, K, r, sigma, DELTA, T, N=500)
    prices     = []

    for deg in degrees:
        p, _ = lsm_put(paths, K, r, T, degree=deg)
        prices.append(p)
        print(f"  Grad {deg}: LSM={p:.4f}  Benchmark={benchmark:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(d) for d in degrees], prices, color="steelblue",
           alpha=0.8, label="LSM-Preis")
    ax.axhline(benchmark, color="crimson", ls="--", lw=2,
               label=f"Binomial Benchmark ({benchmark:.4f})")
    ax.set_xlabel("Polynomgrad der Basisfunktionen")
    ax.set_ylabel("Optionspreis")
    ax.set_title("LSM: Einfluss der Basisfunktionen (American Put)")
    ax.legend()
    ax.set_ylim(min(prices) * 0.97, max(prices) * 1.03)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/05_basis_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. HAUPTPROGRAMM
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  American Options Pricing – LSM (Glasserman Kap. 8.6)")
    print("=" * 65)
    print()

    # ── Parameter laden ──────────────────────────────────────────────────────
    p1, p2 = get_parameters()
    sigma1 = p1["sigma"]
    sigma2 = p2["sigma"]

    # Für die Simulation verwenden wir S0=100 (normiert), damit Ergebnisse
    # mit Glasserman Tabellen vergleichbar sind
    print(f"  Optionsparameter: K={K}  T={T}  r={r}  δ={DELTA}")
    print(f"  σ₁={sigma1:.4f}  σ₂={sigma2:.4f}  Pfade={N_PATHS}")
    print()

    # ── Pfade simulieren ─────────────────────────────────────────────────────
    print("── Simuliere GBM-Pfade ──────────────────────────────────")
    paths1 = simulate_gbm(S0, sigma1, r, DELTA, T, M_STEPS, N_PATHS, SEED)
    paths2 = simulate_gbm(S0, sigma2, r, DELTA, T, M_STEPS, N_PATHS, SEED + 1)
    print(f"  paths1.shape={paths1.shape}  (Pfade × Zeitschritte+1)")
    print()

    # ── American Put (1 Asset) ───────────────────────────────────────────────
    print("── American Put (1 Asset) ──────────────────────────────")
    put_price, ex_t = lsm_put(paths1, K, r, T, degree=3)
    benchmark_put   = binomial_american_put(S0, K, r, sigma1, DELTA, T, N=500)
    print(f"  LSM-Preis:          {put_price:.4f}")
    print(f"  Binomial Benchmark: {benchmark_put:.4f}")
    print(f"  Abweichung:         {abs(put_price - benchmark_put):.4f}")
    print(f"  Ø Ausübungszeitpunkt: {ex_t.mean():.2f}/{M_STEPS}")
    print()

    # ── American Max-Option (2 Assets) ───────────────────────────────────────
    print("── American Max-Option (2 Assets, Glasserman Bsp. 8.6.1) ──")
    max_price, ex_t2, betas = lsm_max_option(
        paths1, paths2, K, r, T, include_h=True)
    # Benchmark aus Glasserman: 13.90 für S1(0)=S2(0)=100
    glasserman_ref = 13.90
    print(f"  LSM-Preis:               {max_price:.4f}")
    print(f"  Glasserman Referenz:     {glasserman_ref:.2f}  (Tabelle 8.1)")
    print(f"  Abweichung:              {abs(max_price - glasserman_ref):.4f}")
    print()

    # ── Sensitivitätsanalyse: Basis-Polynomgrad ───────────────────────────────
    print("── Einfluss der Basisfunktionen (Put) ──────────────────")
    plot_basis_comparison(paths1, K, r, T, sigma1)
    print()

    # ── Konvergenzanalyse ─────────────────────────────────────────────────────
    print("── Konvergenzanalyse ──────────────────────────────────")
    conv = convergence_analysis(sigma1, path_counts=[250, 500, 1000, 2000, 4000])
    print()

    # ── Visualisierungen ──────────────────────────────────────────────────────
    print("── Erstelle Plots ──────────────────────────────────────")
    plot_gbm_paths(paths1, title=f"Simulierte GBM-Pfade  (σ={sigma1:.3f})")
    plot_continuation_value(paths1, K, r, T, i_step=4, degree=3)
    plot_exercise_region_2d(paths1, paths2, K, r, T, i_step=4)
    plot_convergence(conv)

    print()
    print("=" * 65)
    print("  Fertig. Plots gespeichert in /mnt/user-data/outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
