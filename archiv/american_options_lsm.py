"""
==============================================================================
American Options Pricing – Longstaff-Schwartz (LSM)
Basis: Glasserman (2003), Kapitel 8.6
==============================================================================

Verbesserungen gegenüber v1:
  [A1] Out-of-Sample Pricing  – Training- und Pricing-Pfade getrennt
       → behebt In-Sample-Bias, liefert echten Low Estimator (Glasserman 8.34)
  [B1] Antithetische Variablen – halbiert Varianz ohne Mehraufwand
  [B2] Laguerre-Polynome + konsistente Normalisierung über beide Pfad-Sets
  [C]  Kaggle-Daten – automatischer Download ODER manueller CSV-Pfad
  [D]  σ-Sensitivitätsanalyse  – zeigt LSM-Verhalten bei hoher Volatilität

Abhängigkeiten:
  pip install numpy scipy matplotlib pandas kaggle yfinance
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.linalg import lstsq
import warnings, os, pathlib
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Datenquelle ──────────────────────────────────────────────────────────────
USE_REAL_DATA        = True   # yfinance für σ-Kalibrierung
KAGGLE_AUTO_DOWNLOAD = False  # True → kaggle API; False → manuelle CSV

# Kaggle-Dataset (AUTO):  kaggle datasets download -d <DATASET_SLUG>
KAGGLE_DATASET_SLUG  = "kylegraupe/aapl-options-data-2016-2020"
KAGGLE_CSV_FILENAME  = "AAPL_Options.csv"   # Dateiname im ZIP

# Manueller Pfad (falls KAGGLE_AUTO_DOWNLOAD=False und Datei schon vorhanden)
MANUAL_CSV_PATH      = "AAPL_Options.csv"

TICKER_1             = "AAPL"
TICKER_2             = "SPY"
DATA_PERIOD          = "3y"

# ── Optionsparameter (Glasserman 8.6.1 / 8.6.3) ─────────────────────────────
S0        = 100.0   # Anfangskurs (normiert)
K         = 100.0   # Strike
T         = 3.0     # Laufzeit (Jahre)
r         = 0.05    # risikofreier Zinssatz
DELTA     = 0.10    # Dividendenrendite
M_STEPS   = 9       # diskrete Ausübungszeitpunkte
SEED      = 42

# ── Pfadanzahlen ─────────────────────────────────────────────────────────────
N_TRAIN   = 4000    # Pfade für Regressions-Training
N_PRICE   = 4000    # Pfade für Out-of-Sample Pricing

# ── σ-Sensitivitätsanalyse ───────────────────────────────────────────────────
SIGMA_GRID = [0.10, 0.20, 0.40, 0.60, 0.80, 1.20]  # 0.20≈AAPL, 1.20≈Krypto

OUT_DIR   = pathlib.Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1. DATENBESCHAFFUNG
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sigma_yfinance(ticker: str, period: str = "3y") -> float:
    """
    Kalibriert σ (annualisiert) aus historischen Log-Returns via yfinance.
    Fällt auf SIGMA_DEFAULT=0.20 zurück wenn kein Netzwerkzugang.
    """
    try:
        import yfinance as yf
        hist     = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            raise ValueError("Leere Daten")
        lr       = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        sigma    = float(lr.std() * np.sqrt(252))
        mu       = float(lr.mean() * 252)
        S_last   = float(hist["Close"].iloc[-1])
        print(f"  [{ticker}]  σ={sigma:.4f}  μ={mu:.4f}  "
              f"S_last={S_last:.2f}  n={len(hist)} Tage")
        return sigma
    except Exception as e:
        print(f"  [{ticker}] yfinance nicht verfügbar ({e}) → σ=0.20")
        return 0.20


def load_kaggle_options(auto: bool = KAGGLE_AUTO_DOWNLOAD) -> pd.DataFrame | None:
    """
    Lädt historische AAPL-Optionsdaten.

    auto=True  → kaggle API  (benötigt ~/.kaggle/kaggle.json)
    auto=False → liest MANUAL_CSV_PATH

    Kaggle-Setup (einmalig):
        1. kaggle.com → Account → API → "Create New Token" → kaggle.json
        2. Datei nach ~/.kaggle/kaggle.json kopieren
        3. chmod 600 ~/.kaggle/kaggle.json
        4. KAGGLE_AUTO_DOWNLOAD = True setzen

    Rückgabe: DataFrame mit Spalten
        ['strike', 'last_price', 'expiry_days', 'type', 'impliedVolatility']
    oder None wenn Daten nicht verfügbar.
    """
    if auto:
        try:
            import kaggle
            print(f"  Lade Kaggle-Dataset: {KAGGLE_DATASET_SLUG}")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET_SLUG,
                path="./kaggle_data",
                unzip=True,
                quiet=False,
            )
            csv_path = f"./kaggle_data/{KAGGLE_CSV_FILENAME}"
        except Exception as e:
            print(f"  Kaggle-Download fehlgeschlagen: {e}")
            print("  → Setze KAGGLE_AUTO_DOWNLOAD=False und lade manuell.")
            return None
    else:
        csv_path = MANUAL_CSV_PATH
        if not os.path.exists(csv_path):
            print(f"  Kaggle-CSV nicht gefunden: {csv_path}")
            print("  → Lade die Datei manuell von Kaggle herunter und setze MANUAL_CSV_PATH.")
            return None

    # ── Einlesen & Normalisierung ─────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
        print(f"  Kaggle-Daten geladen: {len(df)} Zeilen, Spalten: {list(df.columns)}")

        # Flexibles Spalten-Mapping (verschiedene Kaggle-Datasets haben
        # unterschiedliche Spaltennamen)
        col_map = {
            # Strike
            "Strike":           "strike",
            "strike":           "strike",
            "Strike Price":     "strike",
            # Letzter Preis
            "Last Price":       "last_price",
            "lastPrice":        "last_price",
            "last_price":       "last_price",
            # Tage bis Verfall
            "Expiry":           "expiry_days",
            "expiry":           "expiry_days",
            "DaysToExpiry":     "expiry_days",
            # Typ
            "Type":             "type",
            "type":             "type",
            "optionType":       "type",
            # Implizite Volatilität
            "Implied Volatility": "impliedVolatility",
            "impliedVolatility":  "impliedVolatility",
            "IV":                 "impliedVolatility",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Sicherstellen dass Mindestspalten vorhanden
        required = ["strike", "last_price"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            print(f"  Warnung: Spalten fehlen: {missing}. Überspringe Kaggle-Vergleich.")
            return None

        df = df.dropna(subset=required)
        df["strike"]     = pd.to_numeric(df["strike"],     errors="coerce")
        df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
        df = df.dropna(subset=required)
        return df

    except Exception as e:
        print(f"  Fehler beim Einlesen der CSV: {e}")
        return None


def filter_options(df: pd.DataFrame,
                   opt_type: str = "put",
                   min_price: float = 0.10) -> pd.DataFrame:
    """
    Filtert Put- oder Call-Optionen aus dem Kaggle-DataFrame.
    Entfernt billige (illiquide) Optionen.
    """
    if "type" in df.columns:
        mask = df["type"].str.lower().str.contains(opt_type.lower())
        df   = df[mask]
    df = df[df["last_price"] >= min_price]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  2. GBM-SIMULATION  [B1: Antithetische Variablen]
# ─────────────────────────────────────────────────────────────────────────────

def simulate_gbm(S0: float, sigma: float, r: float, delta: float,
                 T: float, m: int, n_paths: int,
                 antithetic: bool = True,
                 seed: int = None) -> np.ndarray:
    """
    Simuliert GBM-Pfade unter dem risikoneutralen Maß.

        S(t+dt) = S(t) · exp((r − δ − σ²/2)·dt + σ·√dt·Z)

    [B1] Antithetische Variablen:
        Für je n_paths/2 Zufallszahlen Z werden auch −Z simuliert.
        Das halbiert die Varianz des Schätzers fast ohne Mehraufwand.

    Returns:
        paths: (n_paths, m+1)  –  paths[:,0] = S0
    """
    if seed is not None:
        np.random.seed(seed)

    dt    = T / m
    drift = (r - delta - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_h  = np.random.standard_normal((half, m))
        Z    = np.vstack([Z_h, -Z_h])        # gespiegelte Pfade angehängt
    else:
        Z = np.random.standard_normal((n_paths, m))

    log_S = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(log_S)])


# ─────────────────────────────────────────────────────────────────────────────
#  3. BASISFUNKTIONEN  [B2: Laguerre-Polynome + konsistente Normalisierung]
# ─────────────────────────────────────────────────────────────────────────────

def laguerre_basis(S: np.ndarray, degree: int = 4,
                   S_scale: float = None) -> tuple[np.ndarray, float]:
    """
    Laguerre-Polynome als Basisfunktionen (Longstaff-Schwartz [241]).

    L_n(x) via numpy.polynomial.laguerre.lagvander:
        L_0(x) = 1
        L_1(x) = 1 − x
        L_2(x) = 1 − 2x + x²/2
        ...

    [B2] S_scale: Normalisierungskonstante.
        Im Training: wird aus den Trainingsdaten berechnet und zurückgegeben.
        Im Pricing:  wird der aus dem Training gespeicherte Wert übergeben,
                     damit beide Datensätze identisch skaliert sind.

    Returns:
        Phi:    (n, degree+1) Design-Matrix
        S_scale: Normalisierungskonstante (für spätere Verwendung)
    """
    from numpy.polynomial.laguerre import lagvander

    if S_scale is None:
        S_scale = float(S.mean())           # einmalig im Training bestimmen

    x   = S / S_scale                      # dimensionslose Normierung
    Phi = lagvander(x, degree)             # (n, degree+1) Vandermonde-Matrix
    return Phi, S_scale


def basis_max(S1: np.ndarray, S2: np.ndarray,
              s1_scale: float = None, s2_scale: float = None,
              include_h: bool = True) -> tuple[np.ndarray, float, float]:
    """
    Basisfunktionen für die Max-Option (2 Assets), Tabelle 8.1.
    Enthält: 1, S1, S2, S1², S2², S1·S2, [h(S1,S2)]

    [B2] s1_scale, s2_scale: analog zu laguerre_basis.
    """
    if s1_scale is None:
        s1_scale = float(S1.mean())
    if s2_scale is None:
        s2_scale = float(S2.mean())

    s1   = S1 / s1_scale
    s2   = S2 / s2_scale
    cols = [np.ones(len(S1)), s1, s2, s1**2, s2**2, s1 * s2]

    if include_h:
        h = np.maximum(np.maximum(S1, S2) - K, 0.0)
        h_scale = max(h.mean(), 1e-8)
        cols.append(h / h_scale)

    return np.column_stack(cols), s1_scale, s2_scale


# ─────────────────────────────────────────────────────────────────────────────
#  4. LSM-ALGORITHMUS  [A1: Getrennte Training- und Pricing-Pfade]
# ─────────────────────────────────────────────────────────────────────────────

# ── 4a. Trainingsphase: Rückwärts-Induktion → speichert β_i ─────────────────

def lsm_train_put(train_paths: np.ndarray, K: float, r: float, T: float,
                  degree: int = 4) -> dict:
    """
    [A1] TRAININGSPHASE – Rückwärts-Induktion auf train_paths.

    Speichert für jeden Zeitschritt i:
      - beta_i:   Regressionskoeffizienten
      - S_scale_i: Normalisierungskonstante (für konsistente Skalierung)

    Kein Preis wird hier berechnet – das ist der entscheidende Unterschied
    zum naiven In-Sample-Ansatz.

    Returns:
        model: dict mit 'betas', 'scales', 'degree', 'K', 'r', 'T'
    """
    n, m1   = train_paths.shape
    m       = m1 - 1
    dt      = T / m
    df_step = np.exp(-r * dt)

    # Initialisiere Cashflows am Verfallstag
    cashflow   = np.maximum(K - train_paths[:, -1], 0.0)
    exercise_t = np.full(n, m, dtype=int)

    betas  = {}   # beta_i für jeden Schritt
    scales = {}   # S_scale_i für jeden Schritt

    for i in range(m - 1, 0, -1):
        S_i = train_paths[:, i]
        h_i = np.maximum(K - S_i, 0.0)
        itm = h_i > 0

        if itm.sum() < degree + 2:
            cashflow *= df_step
            continue

        # Diskontierte zukünftige Cashflows der ITM-Pfade
        pv = cashflow[itm] * df_step ** (exercise_t[itm] - i)

        # [B2] Laguerre-Basis mit Training-Normalisierung
        Phi, S_scale = laguerre_basis(S_i[itm], degree, S_scale=None)
        beta, _, _, _ = lstsq(Phi, pv)

        betas[i]  = beta
        scales[i] = S_scale

        # Ausübungsentscheidung auf Trainingsdaten (für Cashflow-Update)
        C_hat    = np.maximum(Phi @ beta, 0.0)
        exercise = h_i[itm] >= C_hat
        idx      = np.where(itm)[0]
        cashflow[idx[exercise]]   = h_i[itm][exercise]
        exercise_t[idx[exercise]] = i
        cashflow[idx[~exercise]] *= df_step

    return dict(betas=betas, scales=scales, degree=degree, K=K, r=r, T=T, m=m)


def lsm_train_max(train1: np.ndarray, train2: np.ndarray,
                  K: float, r: float, T: float) -> dict:
    """
    [A1] TRAININGSPHASE für Max-Option (2 Assets).
    Analog zu lsm_train_put.
    """
    n, m1   = train1.shape
    m       = m1 - 1
    dt      = T / m
    df_step = np.exp(-r * dt)

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K, 0.0)

    cashflow   = payoff(train1[:, -1], train2[:, -1])
    exercise_t = np.full(n, m, dtype=int)

    betas, scales1, scales2 = {}, {}, {}

    for i in range(m - 1, 0, -1):
        S1_i = train1[:, i]
        S2_i = train2[:, i]
        h_i  = payoff(S1_i, S2_i)
        itm  = h_i > 0

        if itm.sum() < 10:
            cashflow *= df_step
            continue

        pv  = cashflow[itm] * df_step ** (exercise_t[itm] - i)
        Phi, sc1, sc2 = basis_max(S1_i[itm], S2_i[itm])
        beta, _, _, _ = lstsq(Phi, pv)

        betas[i]  = beta
        scales1[i] = sc1
        scales2[i] = sc2

        C_hat    = np.maximum(Phi @ beta, 0.0)
        exercise = h_i[itm] >= C_hat
        idx      = np.where(itm)[0]
        cashflow[idx[exercise]]   = h_i[itm][exercise]
        exercise_t[idx[exercise]] = i
        cashflow[idx[~exercise]] *= df_step

    return dict(betas=betas, scales1=scales1, scales2=scales2, K=K, r=r, T=T, m=m)


# ── 4b. Pricing-Phase: Vorwärts-Pass auf neuen Pfaden ───────────────────────

def lsm_price_put(model: dict, price_paths: np.ndarray) -> tuple[float, np.ndarray]:
    """
    [A1] PRICING-PHASE – Vorwärts-Pass auf unabhängigen Pfaden.

    Nutzt die im Training gespeicherten β_i und S_scale_i.
    Läuft vorwärts durch die Zeit und übt aus sobald h(S) ≥ Ĉ(S).

    Das ist der echte Low Estimator nach Glasserman (8.33)-(8.34):
    Da die Stopp-Strategie suboptimal sein kann, ist dieser Schätzer
    nach unten verzerrt – aber konsistent und unverzerrt im Limit b→∞.

    Returns:
        price:        Optionspreis-Schätzung
        exercise_idx: (n_paths,) Ausübungszeitpunkt je Pfad
    """
    betas   = model["betas"]
    scales  = model["scales"]
    degree  = model["degree"]
    K_      = model["K"]
    r_      = model["r"]
    T_      = model["T"]
    m       = model["m"]
    dt      = T_ / m
    df_step = np.exp(-r_ * dt)

    n        = price_paths.shape[0]
    exercised = np.zeros(n, dtype=bool)
    payoffs   = np.zeros(n)
    ex_times  = np.full(n, m, dtype=int)

    # Vorwärts durch die Zeitschritte
    for i in range(1, m):
        if i not in betas:
            continue
        active = ~exercised
        if active.sum() == 0:
            break

        S_i = price_paths[active, i]
        h_i = np.maximum(K_ - S_i, 0.0)

        # [B2] Gleiche Normalisierungskonstante wie im Training
        Phi, _ = laguerre_basis(S_i, degree, S_scale=scales[i])
        C_hat  = np.maximum(Phi @ betas[i], 0.0)

        exercise = h_i >= C_hat
        idx_active = np.where(active)[0]

        payoffs[idx_active[exercise]]  = h_i[exercise]
        ex_times[idx_active[exercise]] = i
        exercised[idx_active[exercise]] = True

    # Nicht ausgeübte Pfade: terminaler Payoff
    not_ex = ~exercised
    payoffs[not_ex]  = np.maximum(K_ - price_paths[not_ex, -1], 0.0)
    ex_times[not_ex] = m

    # Diskontierung auf t=0
    discount = df_step ** ex_times
    price    = float(np.mean(payoffs * discount))
    return price, ex_times


def lsm_price_max(model: dict, price1: np.ndarray,
                  price2: np.ndarray) -> tuple[float, np.ndarray]:
    """
    [A1] PRICING-PHASE für Max-Option.
    """
    betas   = model["betas"]
    sc1_map = model["scales1"]
    sc2_map = model["scales2"]
    K_      = model["K"]
    r_      = model["r"]
    T_      = model["T"]
    m       = model["m"]
    dt      = T_ / m
    df_step = np.exp(-r_ * dt)

    def payoff(s1, s2):
        return np.maximum(np.maximum(s1, s2) - K_, 0.0)

    n         = price1.shape[0]
    exercised = np.zeros(n, dtype=bool)
    payoffs   = np.zeros(n)
    ex_times  = np.full(n, m, dtype=int)

    for i in range(1, m):
        if i not in betas:
            continue
        active = ~exercised
        if active.sum() == 0:
            break

        S1_i = price1[active, i]
        S2_i = price2[active, i]
        h_i  = payoff(S1_i, S2_i)

        Phi, _, _ = basis_max(S1_i, S2_i,
                               s1_scale=sc1_map[i],
                               s2_scale=sc2_map[i])
        C_hat  = np.maximum(Phi @ betas[i], 0.0)
        ex     = h_i >= C_hat

        idx = np.where(active)[0]
        payoffs[idx[ex]]   = h_i[ex]
        ex_times[idx[ex]]  = i
        exercised[idx[ex]] = True

    not_ex           = ~exercised
    payoffs[not_ex]  = payoff(price1[not_ex, -1], price2[not_ex, -1])
    ex_times[not_ex] = m

    price = float(np.mean(payoffs * df_step ** ex_times))
    return price, ex_times


# ─────────────────────────────────────────────────────────────────────────────
#  5. BINOMIAL-LATTICE (CRR-Benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def binomial_put(S0: float, K: float, r: float, sigma: float,
                 delta: float, T: float, N: int = 500) -> float:
    """
    CRR-Binomialbaum für American Put.
    N=500 liefert auf 4 Stellen genaue Preise.
    """
    dt  = T / N
    u   = np.exp(sigma * np.sqrt(dt))
    d   = 1.0 / u
    p   = (np.exp((r - delta) * dt) - d) / (u - d)
    df  = np.exp(-r * dt)

    j   = np.arange(N + 1)
    S_T = S0 * u**j * d**(N - j)
    V   = np.maximum(K - S_T, 0.0)

    for step in range(N):
        V   = df * (p * V[1:] + (1 - p) * V[:-1])
        j   = np.arange(len(V))
        S_t = S0 * u**j * d**(len(V) - 1 - j)
        V   = np.maximum(V, np.maximum(K - S_t, 0.0))

    return float(V[0])


# ─────────────────────────────────────────────────────────────────────────────
#  6. VISUALISIERUNGEN
# ─────────────────────────────────────────────────────────────────────────────

STYLE = dict(figsize=(9, 4.5), dpi=130)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → gespeichert: {name}")


def plot_gbm_paths(paths: np.ndarray, sigma: float, n_show: int = 60):
    fig, ax = plt.subplots(**STYLE)
    t = np.linspace(0, T, paths.shape[1])
    for i in range(min(n_show, len(paths))):
        ax.plot(t, paths[i], lw=0.55, alpha=0.35, color="steelblue")
    ax.axhline(K, color="crimson", lw=1.8, ls="--", label=f"Strike K={K}")
    ax.set(xlabel="Zeit (Jahre)", ylabel="Aktienkurs S",
           title=f"Simulierte GBM-Pfade  (σ={sigma:.3f}, antithetisch)")
    ax.legend()
    savefig("01_gbm_paths.png")


def plot_continuation_vs_intrinsic(model: dict, train_paths: np.ndarray,
                                   i_step: int = 4):
    """
    Zeigt Ĉ_i(S) (Continuation Value) vs. h(S) = (K−S)⁺ (Intrinsic Value).
    Ausübungsregion = Bereich wo h ≥ Ĉ.
    """
    if i_step not in model["betas"]:
        print(f"  Warnung: kein β für Schritt {i_step}")
        return

    beta    = model["betas"][i_step]
    S_scale = model["scales"][i_step]
    degree  = model["degree"]

    S_range = np.linspace(55, 145, 400)
    Phi, _  = laguerre_basis(S_range, degree, S_scale=S_scale)
    C_hat   = Phi @ beta
    h_range = np.maximum(K - S_range, 0.0)

    fig, ax = plt.subplots(**STYLE)
    ax.plot(S_range, h_range, color="crimson", lw=2.2,
            label="h(S) = (K−S)⁺  [sofortige Ausübung]")
    ax.plot(S_range, C_hat,  color="steelblue", lw=2.2,
            label=f"$\\hat{{C}}_{{{i_step}}}(S)$  [geschätzte Continuation Value]")
    ax.fill_between(S_range, h_range, np.maximum(C_hat, 0),
                    where=(h_range >= np.maximum(C_hat, 0)),
                    alpha=0.22, color="crimson", label="Ausübungsregion")
    ax.axvline(K, color="gray", ls=":", lw=1.2)
    ax.set(xlim=(55, 145), ylim=(-3, 48),
           xlabel="Aktienkurs S", ylabel="Wert",
           title=f"Continuation Value vs. Intrinsic Value  (Zeitschritt i={i_step})")
    ax.legend()
    savefig("02_continuation_value.png")


def plot_exercise_region_2d(model_max: dict,
                             train1: np.ndarray, train2: np.ndarray,
                             i_step: int = 4):
    """
    Ausübungsregion der Max-Option im (S1,S2)-Raum (vgl. Abb. 8.9 Glasserman).
    """
    if i_step not in model_max["betas"]:
        print(f"  Warnung: kein β für Schritt {i_step}")
        return

    beta = model_max["betas"][i_step]
    sc1  = model_max["scales1"][i_step]
    sc2  = model_max["scales2"][i_step]

    sv  = np.linspace(60, 145, 150)
    S1g, S2g = np.meshgrid(sv, sv)
    S1f, S2f = S1g.flatten(), S2g.flatten()

    h_g      = np.maximum(np.maximum(S1f, S2f) - K, 0.0)
    Phi_g, _, _ = basis_max(S1f, S2f, s1_scale=sc1, s2_scale=sc2)
    C_g      = np.maximum(Phi_g @ beta, 0.0)
    exreg    = ((h_g >= C_g) & (h_g > 0)).reshape(S1g.shape)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=130)
    ax.contourf(S1g, S2g, exreg.astype(float),
                levels=[0.5, 1.5], colors=["#6baed6"], alpha=0.45)
    ax.contour(S1g, S2g, exreg.astype(float),
               levels=[0.5], colors=["steelblue"], linewidths=2)
    ax.plot(sv, sv, "k--", lw=1.4, label="S₁ = S₂  (Symmetrieachse)")
    ax.axhline(K, color="gray", ls=":", lw=1, alpha=0.6)
    ax.axvline(K, color="gray", ls=":", lw=1, alpha=0.6)
    ax.set(xlabel="S₁", ylabel="S₂",
           title=f"Ausübungsregion – American Max-Option  (i={i_step})\n"
                 "[Blau = Ausüben  |  Weiß = Halten]")
    ax.legend(fontsize=9)
    savefig("03_exercise_region_2d.png")


def plot_convergence(sigma: float,
                     path_counts=None):
    """
    LSM-Konvergenz in Pfadanzahl – Out-of-Sample (korrekt) vs. In-Sample.
    Zeigt den Bias-Unterschied direkt.
    """
    if path_counts is None:
        path_counts = [200, 500, 1000, 2000, 4000, 8000]

    true_price  = binomial_put(S0, K, r, sigma, DELTA, T, N=800)
    oos_prices, is_prices = [], []

    print(f"  Benchmark (Binomial): {true_price:.4f}")
    for n in path_counts:
        # Out-of-Sample
        tr  = simulate_gbm(S0, sigma, r, DELTA, T, M_STEPS, n,
                           antithetic=True,  seed=n)
        pr  = simulate_gbm(S0, sigma, r, DELTA, T, M_STEPS, n,
                           antithetic=True,  seed=n + 99999)
        mdl = lsm_train_put(tr, K, r, T, degree=4)
        p_oos, _ = lsm_price_put(mdl, pr)

        # In-Sample (naiv, zum Vergleich)
        p_is, _ = lsm_price_put(lsm_train_put(tr, K, r, T, degree=4), tr)

        oos_prices.append(p_oos)
        is_prices.append(p_is)
        print(f"  N={n:5d}  OoS={p_oos:.4f}  InS={p_is:.4f}  "
              f"Benchmark={true_price:.4f}")

    fig, ax = plt.subplots(**STYLE)
    ax.plot(path_counts, oos_prices, "o-", color="steelblue", lw=2,
            label="Out-of-Sample (Low Estimator, korrekt)")
    ax.plot(path_counts, is_prices,  "s--", color="darkorange", lw=2,
            label="In-Sample (Bias-behaftet)")
    ax.axhline(true_price, color="crimson", ls="--", lw=2,
               label=f"Binomial Benchmark ({true_price:.4f})")
    ax.set(xlabel="Anzahl Pfade (log)", ylabel="Optionspreis",
           title="LSM-Konvergenz: Out-of-Sample vs. In-Sample")
    ax.set_xscale("log")
    ax.legend()
    savefig("04_convergence_oos_vs_is.png")
    return true_price


def plot_sigma_sensitivity(sigmas=None):
    """
    [D] σ-Sensitivitätsanalyse: LSM-Preis und Binomial-Benchmark
    für verschiedene Volatilitäten (AAPL ≈ 0.20, Krypto ≈ 1.20).
    """
    if sigmas is None:
        sigmas = SIGMA_GRID

    lsm_prices, bin_prices = [], []
    for sig in sigmas:
        tr  = simulate_gbm(S0, sig, r, DELTA, T, M_STEPS, N_TRAIN,
                           antithetic=True, seed=SEED)
        pr  = simulate_gbm(S0, sig, r, DELTA, T, M_STEPS, N_PRICE,
                           antithetic=True, seed=SEED + 1)
        mdl = lsm_train_put(tr, K, r, T, degree=4)
        p, _ = lsm_price_put(mdl, pr)
        b    = binomial_put(S0, K, r, sig, DELTA, T, N=500)
        lsm_prices.append(p)
        bin_prices.append(b)
        print(f"  σ={sig:.2f}  LSM={p:.4f}  Binomial={b:.4f}  "
              f"Δ={abs(p-b):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)

    ax = axes[0]
    ax.plot(sigmas, bin_prices, "o-", color="crimson",   lw=2, label="Binomial")
    ax.plot(sigmas, lsm_prices, "s--", color="steelblue", lw=2, label="LSM (OoS)")
    ax.axvline(0.20, color="gray",  ls=":", lw=1.2, label="AAPL ≈ 0.20")
    ax.axvline(1.20, color="black", ls=":", lw=1.2, label="Krypto ≈ 1.20")
    ax.set(xlabel="Volatilität σ", ylabel="Optionspreis",
           title="Optionspreis vs. Volatilität")
    ax.legend(fontsize=8)

    ax = axes[1]
    errors = [abs(l - b) for l, b in zip(lsm_prices, bin_prices)]
    ax.bar([f"{s:.2f}" for s in sigmas], errors, color="steelblue", alpha=0.8)
    ax.set(xlabel="Volatilität σ", ylabel="|LSM − Binomial|",
           title="Absoluter Fehler des LSM-Schätzers")

    plt.tight_layout()
    savefig("05_sigma_sensitivity.png")


def plot_kaggle_comparison(df_options: pd.DataFrame,
                           sigma: float, r_: float = 0.05):
    """
    Vergleicht LSM-Preise mit echten Marktpreisen aus Kaggle-Daten.
    Filtert Put-Optionen mit ähnlicher Laufzeit.
    """
    puts = filter_options(df_options, opt_type="put")
    if puts.empty:
        print("  Keine Put-Optionen in Kaggle-Daten gefunden.")
        return

    market_prices, lsm_prices_list, strikes = [], [], []

    for _, row in puts.iterrows():
        K_row   = float(row["strike"])
        mkt_p   = float(row["last_price"])

        if K_row < 50 or K_row > 200 or mkt_p < 0.5:
            continue

        T_row = float(row.get("expiry_days", 252)) / 252.0
        if T_row <= 0:
            continue

        tr  = simulate_gbm(S0, sigma, r_, 0.0, T_row, max(3, M_STEPS), 2000,
                           antithetic=True, seed=SEED)
        pr  = simulate_gbm(S0, sigma, r_, 0.0, T_row, max(3, M_STEPS), 2000,
                           antithetic=True, seed=SEED + 7)
        mdl = lsm_train_put(tr, K_row, r_, T_row, degree=3)
        p, _ = lsm_price_put(mdl, pr)

        strikes.append(K_row)
        market_prices.append(mkt_p)
        lsm_prices_list.append(p)

        if len(strikes) >= 30:
            break

    if not strikes:
        print("  Zu wenige verwertbare Optionen für Plot.")
        return

    fig, ax = plt.subplots(**STYLE)
    ax.scatter(strikes, market_prices, color="crimson",   s=50,
               label="Marktpreis (Kaggle)", zorder=3)
    ax.scatter(strikes, lsm_prices_list, color="steelblue", s=50, marker="^",
               label="LSM-Preis", zorder=3)
    for s, m_, l in zip(strikes, market_prices, lsm_prices_list):
        ax.plot([s, s], [m_, l], color="gray", lw=0.8, alpha=0.6)
    ax.set(xlabel="Strike K", ylabel="Optionspreis",
           title="LSM vs. Marktpreise  (AAPL Puts, Kaggle-Daten)")
    ax.legend()
    savefig("06_kaggle_market_vs_lsm.png")


# ─────────────────────────────────────────────────────────────────────────────
#  7. HAUPTPROGRAMM
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  American Options Pricing – LSM  (Glasserman 2003, Kap. 8.6)")
    print("=" * 65)

    # ── Volatilität kalibrieren ───────────────────────────────────────────────
    print("\n── Parameterkalibrierung ────────────────────────────────")
    sigma1 = fetch_sigma_yfinance(TICKER_1) if USE_REAL_DATA else 0.20
    sigma2 = fetch_sigma_yfinance(TICKER_2) if USE_REAL_DATA else 0.20
    print(f"  σ₁={sigma1:.4f} ({TICKER_1})  σ₂={sigma2:.4f} ({TICKER_2})")

    # ── Kaggle-Daten laden ────────────────────────────────────────────────────
    print("\n── Kaggle-Optionsdaten ──────────────────────────────────")
    kaggle_df = load_kaggle_options(auto=KAGGLE_AUTO_DOWNLOAD)
    if kaggle_df is not None:
        print(f"  {len(kaggle_df)} Optionskontrakte geladen.")
    else:
        print("  Kaggle-Daten nicht verfügbar – Marktvergleich wird übersprungen.")

    # ── GBM-Pfade simulieren ──────────────────────────────────────────────────
    print("\n── GBM-Simulation  [antithetisch] ───────────────────────")
    train1 = simulate_gbm(S0, sigma1, r, DELTA, T, M_STEPS, N_TRAIN,
                          antithetic=True, seed=SEED)
    price1 = simulate_gbm(S0, sigma1, r, DELTA, T, M_STEPS, N_PRICE,
                          antithetic=True, seed=SEED + 1)
    train2 = simulate_gbm(S0, sigma2, r, DELTA, T, M_STEPS, N_TRAIN,
                          antithetic=True, seed=SEED + 2)
    price2 = simulate_gbm(S0, sigma2, r, DELTA, T, M_STEPS, N_PRICE,
                          antithetic=True, seed=SEED + 3)
    print(f"  Train-Pfade: {train1.shape}  |  Price-Pfade: {price1.shape}")

    # ── American Put  ─────────────────────────────────────────────────────────
    print("\n── American Put (1 Asset, Out-of-Sample) ────────────────")
    model_put       = lsm_train_put(train1, K, r, T, degree=4)
    put_price, ex_t = lsm_price_put(model_put, price1)
    benchmark_put   = binomial_put(S0, K, r, sigma1, DELTA, T, N=500)
    print(f"  LSM (OoS):          {put_price:.4f}")
    print(f"  Binomial Benchmark: {benchmark_put:.4f}")
    print(f"  |Δ|:                {abs(put_price - benchmark_put):.4f}")
    print(f"  Ø Ausübungsschritt: {ex_t.mean():.2f} / {M_STEPS}")

    # ── American Max-Option  ──────────────────────────────────────────────────
    print("\n── American Max-Option (2 Assets, Glasserman 8.6.1) ─────")
    model_max          = lsm_train_max(train1, train2, K, r, T)
    max_price, ex_t2   = lsm_price_max(model_max, price1, price2)
    glasserman_ref     = 13.90
    print(f"  LSM (OoS):           {max_price:.4f}")
    print(f"  Glasserman Ref.:     {glasserman_ref:.2f}  (Tabelle 8.1)")
    print(f"  |Δ|:                 {abs(max_price - glasserman_ref):.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Erstelle Plots ───────────────────────────────────────")
    plot_gbm_paths(train1, sigma1)
    plot_continuation_vs_intrinsic(model_put, train1, i_step=4)
    plot_exercise_region_2d(model_max, train1, train2, i_step=4)
    plot_convergence(sigma1, path_counts=[250, 500, 1000, 2000, 4000, 8000])

    print("\n── σ-Sensitivitätsanalyse ───────────────────────────────")
    plot_sigma_sensitivity(SIGMA_GRID)

    if kaggle_df is not None:
        print("\n── Marktvergleich (Kaggle vs. LSM) ──────────────────────")
        plot_kaggle_comparison(kaggle_df, sigma=sigma1)

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ERGEBNISZUSAMMENFASSUNG")
    print("=" * 65)
    print(f"  American Put   – LSM={put_price:.4f}  Binomial={benchmark_put:.4f}")
    print(f"  American Max   – LSM={max_price:.4f}  Glasserman-Ref={glasserman_ref:.2f}")
    print(f"  Plots gespeichert in: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()