import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class HistoricalDataFetcher:
    def __init__(self, ticker_1: str, ticker_2: str, start_date: str, asset_class: str = "stock", cache_dir: str = "data"):
        """
        Initializes the fetcher.
        :param asset_class: "stock" (uses Yahoo Finance) or "crypto" (uses Binance Public API)
        """
        self.ticker_1 = ticker_1.upper()
        self.ticker_2 = ticker_2.upper()
        self.start_date = start_date
        self.asset_class = asset_class.lower()
        self.cache_dir = cache_dir
        
        # Determine annualization factor based on asset class
        self.trading_days = 365 if self.asset_class == "crypto" else 252
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{self.ticker_1}_{self.ticker_2}_{self.start_date}_{self.asset_class}.csv")

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches adjusted daily closing prices from cache, Yahoo Finance, or Binance.
        """
        if os.path.exists(self.cache_path):
            print(f"Loading {self.asset_class.upper()} data from cache: {self.cache_path}")
            return pd.read_csv(self.cache_path, index_col="Date", parse_dates=True)
            
        print(f"Downloading historical data for {self.ticker_1} and {self.ticker_2}...")
        
        if self.asset_class == "crypto":
            df = self._fetch_binance()
        else:
            df = self._fetch_yahoo()
            
        # Drop any missing values and save to cache
        df = df.dropna()
        df.to_csv(self.cache_path)
        print(f"Data successfully saved to cache: {self.cache_path}")
        
        return df

    def _fetch_yahoo(self) -> pd.DataFrame:
        """
        Fetches data from Yahoo Finance. 
        Uses yf.Ticker().history() to bypass recent yf.download() MultiIndex bugs 
        and missing 'Adj Close' columns.
        """
        # Fetch history directly (this automatically returns adjusted close prices in 'Close')
        t1_data = yf.Ticker(self.ticker_1).history(start=self.start_date)
        t2_data = yf.Ticker(self.ticker_2).history(start=self.start_date)
        
        # Extract the Close column and rename it to the ticker symbol
        s1 = t1_data['Close'].rename(self.ticker_1)
        s2 = t2_data['Close'].rename(self.ticker_2)
        
        # Combine both series into one DataFrame
        df = pd.concat([s1, s2], axis=1).dropna()
        
        # Remove timezone information from the index to keep it clean and compatible
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df

    def _fetch_binance(self) -> pd.DataFrame:
        """Fetches data from Binance Public API (No key required)."""
        def get_binance_klines(symbol):
            # Convert start_date to milliseconds timestamp
            start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
            url = "https://api.binance.com/api/v3/klines"
            
            # Binance limits to 1000 days per request. 1000 days is ~2.7 years.
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ts,
                "limit": 1000 
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            # Index 0: Open time, Index 4: Close price
            dates = [datetime.fromtimestamp(row[0] / 1000).date() for row in data]
            closes = [float(row[4]) for row in data]
            
            return pd.Series(closes, index=pd.to_datetime(dates), name=symbol)

        # Use USDT pairs for Binance (e.g., BTCUSDT, ETHUSDT)
        sym_1 = f"{self.ticker_1}USDT"
        sym_2 = f"{self.ticker_2}USDT"
        
        s1 = get_binance_klines(sym_1)
        s2 = get_binance_klines(sym_2)
        
        df = pd.concat([s1, s2], axis=1)
        df.columns = [self.ticker_1, self.ticker_2]
        df.index.name = "Date"
        return df

    def calculate_historical_parameters(self, df: pd.DataFrame) -> dict:
        """
        Calculates annualized parameters (mu, sigma, rho) using the correct 
        annualization factor (252 for stocks, 365 for crypto).
        """
        print(f"\nCalculating parameters based on {self.trading_days} trading days/year...")
        
        log_returns = np.log(df / df.shift(1)).dropna()
        
        # Annualized Volatility
        sigma_1 = log_returns[self.ticker_1].std() * np.sqrt(self.trading_days)
        sigma_2 = log_returns[self.ticker_2].std() * np.sqrt(self.trading_days)
        
        # Annualized Drift (Itô adjustment included)
        mu_1 = (log_returns[self.ticker_1].mean() * self.trading_days) + (0.5 * sigma_1**2)
        mu_2 = (log_returns[self.ticker_2].mean() * self.trading_days) + (0.5 * sigma_2**2)
        
        # Correlation
        rho = log_returns[self.ticker_1].corr(log_returns[self.ticker_2])
        
        params = {
            "S0_1": float(df[self.ticker_1].iloc[-1]),
            "S0_2": float(df[self.ticker_2].iloc[-1]),
            "mu_1": float(mu_1),
            "mu_2": float(mu_2),
            "sigma_1": float(sigma_1),
            "sigma_2": float(sigma_2),
            "rho": float(rho),
            "asset_class": self.asset_class
        }
        
        for key, value in params.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        return params

if __name__ == "__main__":
    # --- Test 1: Traditional Tech Stocks (Yahoo Finance) ---
    print("=== TEST 1: TECH STOCKS ===")
    tech_fetcher = HistoricalDataFetcher(
        ticker_1="AAPL", ticker_2="MSFT", start_date="2022-01-01", asset_class="stock"
    )
    tech_df = tech_fetcher.fetch_data()
    tech_params = tech_fetcher.calculate_historical_parameters(tech_df)
    
    # --- Test 2: Crypto (Binance API) ---
    print("\n=== TEST 2: CRYPTO ===")
    crypto_fetcher = HistoricalDataFetcher(
        ticker_1="BTC", ticker_2="ETH", start_date="2022-01-01", asset_class="crypto"
    )
    crypto_df = crypto_fetcher.fetch_data()
    crypto_params = crypto_fetcher.calculate_historical_parameters(crypto_df)