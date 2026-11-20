"""
Historical prices from Yahoo Finance and calibration of the volatilities and the correlation of two
assets. Both asset classes (stocks, crypto) use the same source and the same fixed time window.

Only sigma_1, sigma_2 and rho are estimated. Historical log-returns are observed under the
real-world measure P; their drift is not used anywhere, because pricing uses the risk-neutral
drift r - delta. Volatility and correlation are the same under P and Q for a GBM (Girsanov changes
only the drift), so they can be estimated from historical data.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Fixed calibration window (inclusive), identical for all asset classes.
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"


class HistoricalDataFetcher:
    """
    Daily adjusted closing prices of two Yahoo Finance tickers in a fixed window
    [start_date, end_date] (both inclusive), cached as CSV in data/.

    Annualisation: 252 trading days per year for stocks, 365 for crypto (traded every day).
    """

    def __init__(self, ticker_1: str, ticker_2: str, asset_class: str = "stock",
                 start_date: str = START_DATE, end_date: str = END_DATE,
                 cache_dir: str = "data") -> None:
        if asset_class not in ("stock", "crypto"):
            raise ValueError("asset_class must be 'stock' or 'crypto'")
        self.ticker_1 = ticker_1.upper()
        self.ticker_2 = ticker_2.upper()
        self.asset_class = asset_class
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.periods_per_year = 365 if asset_class == "crypto" else 252
        self.cache_path = os.path.join(
            cache_dir, f"yahoo_{self.ticker_1}_{self.ticker_2}_{start_date}_{end_date}.csv")

    def fetch_data(self) -> pd.DataFrame:
        """Prices from the cache if present, otherwise downloaded from Yahoo Finance and cached."""
        if os.path.exists(self.cache_path):
            print(f"Loading data from cache: {self.cache_path}")
            return pd.read_csv(self.cache_path, index_col="Date", parse_dates=True)

        import yfinance as yf  # imported lazily: only needed when the cache is missing

        print(f"Downloading {self.ticker_1}, {self.ticker_2} from Yahoo Finance "
              f"({self.start_date} to {self.end_date})...")
        # yfinance treats `end` as exclusive, so request one day more.
        end_excl = (date.fromisoformat(self.end_date) + timedelta(days=1)).isoformat()
        series = []
        for tk in (self.ticker_1, self.ticker_2):
            hist = yf.Ticker(tk).history(start=self.start_date, end=end_excl, auto_adjust=True)
            if hist.empty:
                raise RuntimeError(f"no data returned for {tk}")
            s = hist["Close"].rename(tk)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = s.index.normalize()
            series.append(s)
        df = pd.concat(series, axis=1).dropna()
        df.index.name = "Date"
        os.makedirs(self.cache_dir, exist_ok=True)
        df.to_csv(self.cache_path)
        print(f"Saved to cache: {self.cache_path}")
        return df

    def calibrate(self, df: pd.DataFrame) -> dict:
        """
        Annualised volatilities (sample std of daily log-returns times sqrt(periods per year))
        and the correlation of the daily log-returns. No drift is estimated.
        """
        log_ret = np.log(df / df.shift(1)).dropna()
        a = np.sqrt(self.periods_per_year)
        params = {
            "sigma_1": float(log_ret[self.ticker_1].std(ddof=1) * a),
            "sigma_2": float(log_ret[self.ticker_2].std(ddof=1) * a),
            "rho": float(log_ret[self.ticker_1].corr(log_ret[self.ticker_2])),
            "n_returns": int(len(log_ret)),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "periods_per_year": self.periods_per_year,
        }
        print(f"{self.ticker_1}/{self.ticker_2}: sigma_1 = {params['sigma_1']:.4f}, "
              f"sigma_2 = {params['sigma_2']:.4f}, rho = {params['rho']:.4f} "
              f"({params['n_returns']} daily returns, {params['first_date']} to "
              f"{params['last_date']}, {self.periods_per_year} periods/year)")
        return params


if __name__ == "__main__":  # python -m maxcall.data
    for t1, t2, cls in (("AAPL", "MSFT", "stock"), ("BTC-USD", "ETH-USD", "crypto")):
        f = HistoricalDataFetcher(t1, t2, cls)
        f.calibrate(f.fetch_data())
