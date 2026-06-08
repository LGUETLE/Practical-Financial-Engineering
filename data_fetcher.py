import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class HistoricalDataFetcher:
    def __init__(self, ticker_1: str, ticker_2: str, start_date: str, end_date: str = None, cache_dir: str = "data"):
        """
        Initializes the data fetcher for two assets.
        """
        self.ticker_1 = ticker_1
        self.ticker_2 = ticker_2
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{ticker_1}_{ticker_2}_{self.start_date}_to_{self.end_date}.csv")

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches adjusted close prices. Uses local CSV cache if available to save API calls.
        """
        if os.path.exists(self.cache_path):
            print(f"Loading historical data from cache: {self.cache_path}")
            df = pd.read_csv(self.cache_path, index_col="Date", parse_dates=True)
            return df
            
        print(f"Downloading data from Yahoo Finance for {self.ticker_1} and {self.ticker_2}...")
        data = yf.download([self.ticker_1, self.ticker_2], start=self.start_date, end=self.end_date)
        
        # Extract Adjusted Close prices
        df = data['Adj Close'].dropna()
        
        # Ensure correct column order
        df = df[[self.ticker_1, self.ticker_2]]
        
        # Save to cache
        df.to_csv(self.cache_path)
        print(f"Data saved to cache: {self.cache_path}")
        
        return df

    def calculate_historical_parameters(self, df: pd.DataFrame, trading_days_per_year: int = 252) -> dict:
        """
        Calculates annualized drift (mu), volatility (sigma), and correlation (rho) 
        from daily logarithmic returns.
        """
        print("Calculating historical parameters from log returns...")
        
        # Calculate daily logarithmic returns
        log_returns = np.log(df / df.shift(1)).dropna()
        
        # 1. Annualized Volatility (Standardabweichung der Log-Returns * sqrt(252))
        sigma_1 = log_returns[self.ticker_1].std() * np.sqrt(trading_days_per_year)
        sigma_2 = log_returns[self.ticker_2].std() * np.sqrt(trading_days_per_year)
        
        # 2. Annualized Drift (Mean der Log-Returns * 252 + 0.5 * Varianz)
        # Itô's Lemma adjustment for geometric brownian motion drift
        mu_1 = (log_returns[self.ticker_1].mean() * trading_days_per_year) + (0.5 * sigma_1**2)
        mu_2 = (log_returns[self.ticker_2].mean() * trading_days_per_year) + (0.5 * sigma_2**2)
        
        # 3. Empirical Correlation
        rho = log_returns[self.ticker_1].corr(log_returns[self.ticker_2])
        
        params = {
            "S0_1": float(df[self.ticker_1].iloc[-1]), # Letzter verfügbarer Kurs als S0
            "S0_2": float(df[self.ticker_2].iloc[-1]),
            "mu_1": float(mu_1),
            "mu_2": float(mu_2),
            "sigma_1": float(sigma_1),
            "sigma_2": float(sigma_2),
            "rho": float(rho)
        }
        
        print("\n--- Extracted Market Parameters ---")
        for key, value in params.items():
            print(f"{key}: {value:.4f}")
            
        return params

if __name__ == "__main__":
    # Test the Fetcher with AAPL and MSFT over the last 5 years
    fetcher = HistoricalDataFetcher(
        ticker_1="AAPL", 
        ticker_2="MSFT", 
        start_date="2021-01-01"
    )
    
    price_df = fetcher.fetch_data()
    market_params = fetcher.calculate_historical_parameters(price_df)